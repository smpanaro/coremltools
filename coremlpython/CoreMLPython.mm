// Copyright (c) 2021, Apple Inc. All rights reserved.
//
// Use of this source code is governed by a BSD-3-clause license that can be
// found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

#import <CoreML/CoreML.h>
#import "CoreMLPythonArray.h"
#import "CoreMLPython.h"
#import "CoreMLPythonUtils.h"
#import "Globals.hpp"
#import "Utils.hpp"
#import <AvailabilityMacros.h>
#import <fstream>
#import <vector>

#import <os/log.h>
#import <os/signpost.h>

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wmissing-prototypes"

#if ! __has_feature(objc_arc)
#error "ARC is off"
#endif

#ifndef BUILT_WITH_MACOS13_SDK
#define BUILT_WITH_MACOS13_SDK (MAC_OS_X_VERSION_MAX_ALLOWED >= 130000)
#endif

namespace py = pybind11;

using namespace CoreML::Python;

bool usingMacOS13OrHigher() {
    // MLProgram class was introduced in macOS 13.
    return (NSProtocolFromString(@"MLProgram") != nil);
}

bool isCompiledModelPath(const std::string& path) {
    const std::string fileExtension = ".mlmodelc";

    size_t start = path.length() - fileExtension.length();
    if (path.back() == '/') {
        start--;
    }
    const std::string match = path.substr(start, fileExtension.length());

    return (match == fileExtension);
}

Model::~Model() {
    @autoreleasepool {
        NSFileManager *fileManager = [NSFileManager defaultManager];
        if (compiledUrl != nil and m_deleteCompiledModelOnExit) {
            [fileManager removeItemAtURL:compiledUrl error:NULL];
        }
    }
}

Model::Model(const std::string& urlStr, const std::string& computeUnits) {
    @autoreleasepool {
        NSError *error = nil;

        if (! isCompiledModelPath(urlStr)) {
            // Compile the model
            NSURL *specUrl = Utils::stringToNSURL(urlStr);

            // Swallow output for the very verbose coremlcompiler
            int stdoutBack = dup(STDOUT_FILENO);
            int devnull = open("/dev/null", O_WRONLY);
            dup2(devnull, STDOUT_FILENO);

            // Compile the model
            compiledUrl = [MLModel compileModelAtURL:specUrl error:&error];
            m_deleteCompiledModelOnExit = true;

            // Close all the file descriptors and revert back to normal
            dup2(stdoutBack, STDOUT_FILENO);
            close(devnull);
            close(stdoutBack);

            // Translate into a type that pybind11 can bridge to Python
            if (error != nil) {
                std::stringstream errmsg;
                errmsg << "Error compiling model: \"";
                errmsg << error.localizedDescription.UTF8String;
                errmsg << "\".";
                throw std::runtime_error(errmsg.str());
            }
        } else {
            m_deleteCompiledModelOnExit = false;  // Don't delete user specified file
            compiledUrl = Utils::stringToNSURL(urlStr);
        }

        // Set compute unit
        MLModelConfiguration *configuration = [MLModelConfiguration new];
        if (computeUnits == "CPU_ONLY") {
            configuration.computeUnits = MLComputeUnitsCPUOnly;
        } else if (computeUnits == "CPU_AND_GPU") {
            configuration.computeUnits = MLComputeUnitsCPUAndGPU;
        } else if (computeUnits == "CPU_AND_NE") {
            if (usingMacOS13OrHigher()) {
#if BUILT_WITH_MACOS13_SDK
                configuration.computeUnits = MLComputeUnitsCPUAndNeuralEngine;
#endif // BUILT_WITH_MACOS13_SDK
            } else {
                throw std::runtime_error("CPU_AND_NE is only available on macOS >= 13.0");
            }
        } else {
            assert(computeUnits == "ALL");
            configuration.computeUnits = MLComputeUnitsAll;
        }

        outputCache = [NSMutableDictionary new];

        // Create MLModel
        m_model = [MLModel modelWithContentsOfURL:compiledUrl configuration:configuration error:&error];
        Utils::handleError(error);
    }
}

py::dict Model::predict(const py::dict& input, const std::optional<py::dict>& inputOutputKeyMapping) const {
    @autoreleasepool {
        os_log_t log = Utils::default_log();
        NSError *error = nil;

        // Populate input features from the cache.
        NSDictionary<NSString *, NSString *> *nsKeyMapping = nil;
        if (inputOutputKeyMapping.has_value()) {
            nsKeyMapping = Utils::convertStringDictToObjC(inputOutputKeyMapping.value());
        }
        NSMutableDictionary<NSString *, MLMultiArray *> *extraFeatures = [NSMutableDictionary new];
        [nsKeyMapping enumerateKeysAndObjectsUsingBlock:^(NSString * _Nonnull inputKey, NSString * _Nonnull outputKey, BOOL * _Nonnull stop) {
            if (outputCache[outputKey] != nil) {
                extraFeatures[inputKey] = outputCache[outputKey];
            }
        }];

        os_signpost_id_t inputSignpostId = os_signpost_id_generate(log);
        os_signpost_interval_begin(log, inputSignpostId, "Convert", "Convert Input");
        MLDictionaryFeatureProvider *inFeatures = Utils::dictToFeatures(input, extraFeatures, &error);
        Utils::handleError(error);
        os_signpost_interval_end(log, inputSignpostId, "Convert");

        os_signpost_id_t predictionSignpostId = os_signpost_id_generate(log);
        os_signpost_interval_begin(log, predictionSignpostId, "Predict");
        id<MLFeatureProvider> outFeatures = [m_model predictionFromFeatures:static_cast<MLDictionaryFeatureProvider * _Nonnull>(inFeatures)
                                                                      error:&error];
        Utils::handleError(error);
        os_signpost_interval_end(log, predictionSignpostId, "Predict");

        // Update cache with new output values.
        NSMutableSet<NSString *> *cachedOutputKeys = [NSMutableSet new];
        [nsKeyMapping enumerateKeysAndObjectsUsingBlock:^(NSString * _Nonnull inputKey, NSString * _Nonnull outputKey, BOOL * _Nonnull stop) {
            MLMultiArray *outFeature = [[outFeatures featureValueForName:outputKey] multiArrayValue];
            if (outFeature != nil) {
                outputCache[outputKey] = outFeature;
                [cachedOutputKeys addObject:outputKey];
            }
        }];

        os_signpost_id_t outputSignpostId = os_signpost_id_generate(log);
        os_signpost_interval_begin(log, outputSignpostId, "Convert", "Convert Output");
        py::dict res = Utils::featuresToDict(outFeatures, cachedOutputKeys);
        os_signpost_interval_end(log, outputSignpostId, "Convert");
        return res;
    }
}


py::list Model::batchPredict(const py::list& batch) const {
  @autoreleasepool {
      NSError* error = nil;

      // Convert input to a BatchProvider
      NSMutableArray* array = [[NSMutableArray alloc] initWithCapacity: batch.size()];
      for(int i = 0; i < batch.size(); i++) {
        MLDictionaryFeatureProvider* cur = Utils::dictToFeatures(batch[i], @{}, &error);
        Utils::handleError(error);
        [array addObject: cur];
      }
      MLArrayBatchProvider* batchProvider = [[MLArrayBatchProvider alloc] initWithFeatureProviderArray: array];

      // Get predictions
      MLArrayBatchProvider* predictions = (MLArrayBatchProvider*)[m_model predictionsFromBatch:batchProvider
                                                                                         error:&error];
      Utils::handleError(error);

      // Convert predictions to output
      py::list ret;
      for (int i = 0; i < predictions.array.count; i++) {
        ret.append(Utils::featuresToDict(predictions.array[i], nil));
      }
      return ret;
  }
}


py::str Model::getCompiledModelPath() const {
    return [this->compiledUrl.path UTF8String];
}


py::bytes Model::autoSetSpecificationVersion(const py::bytes& modelBytes) {

    CoreML::Specification::Model model;
    std::istringstream modelIn(static_cast<std::string>(modelBytes), std::ios::binary);
    CoreML::loadSpecification<Specification::Model>(model, modelIn);
    model.set_specificationversion(CoreML::MLMODEL_SPECIFICATION_VERSION_NEWEST);
    // always try to downgrade the specification version to the
    // minimal version that supports everything in this mlmodel
    CoreML::downgradeSpecificationVersion(&model);
    std::ostringstream modelOut;
    saveSpecification(model, modelOut);
    return static_cast<py::bytes>(modelOut.str());

}


py::str Model::compileModel(const std::string& urlStr) {
    @autoreleasepool {
        NSError* error = nil;

        NSURL* specUrl = Utils::stringToNSURL(urlStr);
        NSURL* compiledUrl = [MLModel compileModelAtURL:specUrl error:&error];

        Utils::handleError(error);
        return [compiledUrl.path UTF8String];
    }
}


int32_t Model::maximumSupportedSpecificationVersion() {
    return CoreML::MLMODEL_SPECIFICATION_VERSION_NEWEST;
}


/*
 *
 * bindings
 *
 */

PYBIND11_PLUGIN(libcoremlpython) {
    py::module m("libcoremlpython", "CoreML.Framework Python bindings");

    py::class_<Model>(m, "_MLModelProxy")
        .def(py::init<const std::string&, const std::string&>())
        .def("predict", &Model::predict,
            py::arg("features"), py::arg("input_output_key_mapping") = py::none())
        .def("batchPredict", &Model::batchPredict)
        .def("get_compiled_model_path", &Model::getCompiledModelPath)
        .def_static("auto_set_specification_version", &Model::autoSetSpecificationVersion)
        .def_static("maximum_supported_specification_version", &Model::maximumSupportedSpecificationVersion)
        .def_static("compileModel", &Model::compileModel);

    return m.ptr();
}

#pragma clang diagnostic pop
