// Copyright (c) 2021, Apple Inc. All rights reserved.
//
// Use of this source code is governed by a BSD-3-clause license that can be
// found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause
#include <string>

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wexit-time-destructors"
#pragma clang diagnostic ignored "-Wdocumentation"
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <os/availability.h>
#pragma clang diagnostic pop

#import <Foundation/Foundation.h>
#import <CoreML/CoreML.h>
#import <os/log.h>
#import "LayerShapeConstraints.hpp"

namespace py = pybind11;

namespace CoreML {
    namespace Python {
        namespace Utils {

            NSURL * stringToNSURL(const std::string& str);
            void handleError(NSError *error);
            os_log_t default_log();
            os_log_t dynamic_tracing_log();

            // python -> objc
            MLDictionaryFeatureProvider *
            dictToFeatures(const py::dict &dict, NSDictionary<NSString *, NSObject *> *extraFeatures, NSError **error);
            MLFeatureValue *convertValueToObjC(const py::handle &handle);
            NSDictionary<NSString *, NSString *> *convertStringDictToObjC(const py::dict &dict);

            // objc -> cpp
            std::vector<size_t> convertNSArrayToCpp(NSArray<NSNumber *> *array);
            NSArray<NSNumber *>* convertCppArrayToObjC(const std::vector<size_t>& array);

            // objc -> python
            py::dict featuresToDict(id<MLFeatureProvider> features, NSSet<NSString *> *skipFeatures);
            py::object convertValueToPython(MLFeatureValue *value);
            py::object convertArrayValueToPython(MLMultiArray *value);
            py::object convertDictionaryValueToPython(NSDictionary<NSObject *,NSNumber *> * value);
            py::object convertImageValueToPython(CVPixelBufferRef value);
            py::object convertSequenceValueToPython(MLSequence *seq) API_AVAILABLE(macos(10.14));
        }
    }
}

// A custom type that is the same size as a numpy float16.
struct np_float16_t {
    uint16_t x;
};

namespace pybind11 {
    namespace detail {
        template <>
        struct npy_format_descriptor<np_float16_t> {
            static constexpr auto name = _("float16");
            static pybind11::dtype dtype()
            {
                handle ptr = npy_api::get().PyArray_DescrFromType_(23); /* import numpy as np; print(np.dtype(np.float16).num */
                return reinterpret_borrow<pybind11::dtype>(ptr);
            }
            static std::string format()
            {
                return "e";
            }
        };
    }
}
