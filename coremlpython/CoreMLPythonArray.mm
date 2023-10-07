#import "CoreMLPythonArray.h"

#import <Accelerate/Accelerate.h>

@implementation PybindCompatibleArray

+ (MLMultiArrayDataType)dataTypeOf:(py::array)array {
    const auto& dt = array.dtype();
    char kind = dt.kind();
    size_t itemsize = dt.itemsize();
    
    if(kind == 'i' && itemsize == 4) {
        return MLMultiArrayDataTypeInt32;
    } else if(kind == 'f' && itemsize == 2) {
        return MLMultiArrayDataTypeFloat16;
    } else if(kind == 'f' && itemsize == 4) {
        return MLMultiArrayDataTypeFloat32;
    } else if( (kind == 'f' || kind == 'd') && itemsize == 8) {
        return MLMultiArrayDataTypeDouble;
    }
    
    throw std::runtime_error("Unsupported array type: " + std::to_string(kind) + " with itemsize = " + std::to_string(itemsize));
}

+ (NSArray<NSNumber *> *)shapeOf:(py::array)array {
    NSMutableArray<NSNumber *> *ret = [[NSMutableArray alloc] init];
    for (size_t i=0; i<array.ndim(); i++) {
        [ret addObject:[NSNumber numberWithUnsignedLongLong:array.shape(i)]];
    }
    return ret;
}

+ (NSArray<NSNumber *> *)stridesOf:(py::array)array {
    // numpy strides is in bytes.
    // this type must return number of ELEMENTS! (as per mlkit)
    
    NSMutableArray<NSNumber *> *ret = [[NSMutableArray alloc] init];
    for (size_t i=0; i<array.ndim(); i++) {
        size_t stride = array.strides(i) / array.itemsize();
        [ret addObject:[NSNumber numberWithUnsignedLongLong:stride]];
    }
    return ret;
}

// Returns a CVPixelBufferRef backed by an IOSurface that can be used to initialize a MLMultiArray.
// This avoids memory copies and provides significant performance benefits for large arrays.
+ (CVPixelBufferRef)float16PixelBufferBackedArrayFrom:(py::array)array withShape:(NSArray<NSNumber *> *)shape {
    // Per the header doc for initWithPixelBuffer.
    size_t width = shape[shape.count - 1].unsignedLongLongValue;
    size_t height = 1;
    for (int i=0; i<shape.count - 1; i++) {
        height *= shape[i].unsignedLongLongValue;
    }

    CVPixelBufferRef pixelBuffer = NULL;
    NSDictionary* pixelBufferAttributes = @{
        (id)kCVPixelBufferIOSurfacePropertiesKey: @{}
    };

    // Can't use CVPixelBufferCreateWithBytes, because it doesn't support kCVPixelBufferIOSurfacePropertiesKey.
    // https://developer.apple.com/library/archive/qa/qa1781/_index.html
    CVReturn status = CVPixelBufferCreate(kCFAllocatorDefault,
        width,
        height,
        kCVPixelFormatType_OneComponent16Half,
        (__bridge CFDictionaryRef)pixelBufferAttributes,
        &pixelBuffer);

    if (status != kCVReturnSuccess) {
        std::stringstream msg;
        msg << "Got unexpected return code " << status << " from CVPixelBufferCreate.";
        py::print(msg.str());
        throw std::runtime_error(msg.str());
    }

    status = CVPixelBufferLockBaseAddress(pixelBuffer, 0);
    if (status != kCVReturnSuccess) {
        std::stringstream msg;
        msg << "Got unexpected return code " << status << " from CVPixelBufferLockBaseAddress.";
        py::print(msg.str());
        throw std::runtime_error(msg.str());
    }
    void *baseAddress = CVPixelBufferGetBaseAddress(pixelBuffer);
    assert(baseAddress != nullptr);
    assert(!CVPixelBufferIsPlanar(pixelBuffer));

    // CVPixelBuffer can be padded so it is aligned. In that case, memcpy isn't safe.
    // https://stackoverflow.com/questions/46879895/byte-per-row-is-wrong-when-creating-a-cvpixelbuffer-with-width-multiple-of-90V>
    // assert(CVPixelBufferGetBytesPerRow(pixelBuffer) == width * sizeof(np_float16_t));
    // memcpy(baseAddress, array.mutable_data(), array.nbytes());

    // vImage doesn't seem to provided any performance benefit over memcpy,
    // but it does handle the rowBytes padding automatically.
    vImage_Buffer srcBuffer;
    memset(&srcBuffer, 0, sizeof(srcBuffer));
    srcBuffer.data = array.mutable_data();
    srcBuffer.width = width;
    srcBuffer.height = height;
    srcBuffer.rowBytes = array.itemsize() * width;

    vImage_Buffer dstBuffer;
    memset(&dstBuffer, 0, sizeof(dstBuffer));
    dstBuffer.data = baseAddress;
    dstBuffer.width = width;
    dstBuffer.height = height;
    dstBuffer.rowBytes = CVPixelBufferGetBytesPerRow(pixelBuffer);

    vImageCopyBuffer(&srcBuffer, &dstBuffer, sizeof(np_float16_t), 0);

    status = CVPixelBufferUnlockBaseAddress(pixelBuffer, 0);
    if (status != kCVReturnSuccess) {
        std::stringstream msg;
        msg << "Got unexpected return code " << status << " from CVPixelBufferUnlockBaseAddress.";
        py::print(msg.str());
        throw std::runtime_error(msg.str());
    }

    return pixelBuffer;
}

- (PybindCompatibleArray *)initWithArray:(py::array)array {
    MLMultiArrayDataType dataType = [self.class dataTypeOf:array];
    NSArray<NSNumber *> *shape = [self.class shapeOf:array];

    if (dataType == MLMultiArrayDataTypeFloat16) {
        CVPixelBufferRef pixelBuffer = [self.class float16PixelBufferBackedArrayFrom:array
                                                                           withShape:shape];
        self = [super initWithPixelBuffer:pixelBuffer shape:shape];
    } else {
        self = [super initWithDataPointer:array.mutable_data()
                                    shape:shape
                                 dataType:dataType
                                  strides:[self.class stridesOf:array]
                              deallocator:nil
                                    error:nil];
    }

    if (self) {
        m_array = array;
    }
    return self;
}

@end
