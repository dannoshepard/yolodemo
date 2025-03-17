import Vision
import CoreML
import UIKit

class YOLOProcessor {
    static let shared = YOLOProcessor()
    
    private var visionModel: VNCoreMLModel?
    private var isModelLoaded = false
    
    // Screen dimensions from UIKit
    private var screenWidth: CGFloat = UIScreen.main.bounds.width
    private var screenHeight: CGFloat = UIScreen.main.bounds.height
    
    // Update screen dimensions if needed (e.g. on rotation)
    func updateScreenDimensions(width: CGFloat, height: CGFloat) {
        screenWidth = width
        screenHeight = height
        print("Screen dimensions updated: \(screenWidth)x\(screenHeight)")
    }
    
    // Configurable parameters
    struct Config {
        var confidenceThreshold: Float = 0.25
        var iouThreshold: Float = 0.45
        var minBoxSize: Float = 10  // minimum box dimension in pixels
        var maxBoxSize: Float = 600 // maximum box dimension in pixels
    }
    
    var config = Config()
    
    private init() {
        setupModel()
    }
    
    private func setupModel() {
        do {
            print("Loading YOLO model...")
            let config = MLModelConfiguration()
            config.computeUnits = .all
            
            let model = try yolo11n(configuration: config)
            print("Model input description: \(model.model.modelDescription.inputDescriptionsByName)")
            print("Model output description: \(model.model.modelDescription.outputDescriptionsByName)")
            
            if let inputDesc = model.model.modelDescription.inputDescriptionsByName.first?.value {
                print("Input dimensions: \(inputDesc.imageConstraint?.pixelsHigh ?? 0) x \(inputDesc.imageConstraint?.pixelsWide ?? 0)")
                print("Input type: \(inputDesc.type)")
            }
            
            visionModel = try VNCoreMLModel(for: model.model)
            isModelLoaded = true
            print("YOLO model loaded successfully")
        } catch {
            print("Failed to load YOLO model: \(error)")
            isModelLoaded = false
        }
    }
    
    private var currentBuffer: CVPixelBuffer!
    
    func detect(in pixelBuffer: CVPixelBuffer) async throws -> [Detection] {
        guard isModelLoaded, let model = visionModel else {
            print("Model not loaded, cannot perform detection")
            throw NSError(domain: "YOLOProcessor", code: -1, userInfo: [NSLocalizedDescriptionKey: "Model not loaded"])
        }
        
        let width = CVPixelBufferGetWidth(pixelBuffer)
        let height = CVPixelBufferGetHeight(pixelBuffer)
        print("Input image size: \(width)x\(height)")
        
        return try await withCheckedThrowingContinuation { continuation in
            let request = VNCoreMLRequest(model: model) { request, error in
                if let error = error {
                    print("Vision ML Request error: \(error)")
                    continuation.resume(throwing: error)
                    return
                }
                
                guard let results = request.results as? [VNCoreMLFeatureValueObservation],
                      let firstResult = results.first,
                      let featureValue = firstResult.featureValue as? MLFeatureValue,
                      let multiArray = featureValue.multiArrayValue else {
                    print("Could not get feature value from results")
                    continuation.resume(returning: [])
                    return
                }

                let detections = self.processYOLOOutput(multiArray)
                continuation.resume(returning: detections)
            }
            
            // Scale to 640x640
            request.imageCropAndScaleOption = .scaleFit
            request.usesCPUOnly = false
            
            do {
                let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, options: [:])
                try handler.perform([request])
            } catch {
                print("Vision request failed: \(error)")
                continuation.resume(throwing: error)
            }
        }
    }
    
    private func processYOLOOutput(_ multiArray: MLMultiArray) -> [Detection] {
        let numClasses = 80
        let numBoxes = 8400
        
        var candidateDetections: [(Detection, Float)] = []
        
        // Assuming shape is [1, 84, 8400] where 84 = 4(box) + 80(classes)
        for box in 0..<numBoxes {
            if let detection = processBox(box: box, multiArray: multiArray, numClasses: numClasses) {
                candidateDetections.append(detection)
            }
        }

        // Sort by confidence
        candidateDetections.sort { $0.1 > $1.1 }
        
        // Apply NMS
        var finalDetections: [Detection] = []
        while !candidateDetections.isEmpty {
            let current = candidateDetections.removeFirst()
            finalDetections.append(current.0)
            
            // Filter out overlapping boxes
            candidateDetections = candidateDetections.filter { candidate in
                let iou = calculateIOU(current.0.boundingBox, candidate.0.boundingBox)
                return iou < config.iouThreshold
            }
        }
        
        return finalDetections
    }
    
    private func processBox(box: Int, multiArray: MLMultiArray, numClasses: Int) -> (Detection, Float)? {
        // Get coordinates in xywh format (center-x, center-y, width, height)
        let centerX = Float(truncating: multiArray[[0, 0, box] as [NSNumber]])  // center x
        let centerY = Float(truncating: multiArray[[0, 1, box] as [NSNumber]]) // center y
        let width = Float(truncating: multiArray[[0, 2, box] as [NSNumber]])  // width
        let height = Float(truncating: multiArray[[0, 3, box] as [NSNumber]]) // height

        
        // Skip invalid boxes
        guard width > 0 && height > 0 else {
            return nil
        }
        
        // Skip boxes that are too small or too large in pixel space
        let minSize = Float(config.minBoxSize)
        let maxSize = Float(config.maxBoxSize)
        guard width >= minSize && height >= minSize && width <= maxSize && height <= maxSize else {
            return nil
        }
        
        // Find the highest class score and index
        let (maxScore, maxClassIndex) = findBestClass(box: box, multiArray: multiArray, numClasses: numClasses)
        
        // Only keep detections above confidence threshold
        guard maxScore > config.confidenceThreshold else {
            return nil
        }

        print("\nBox #\(box) Raw Data:")
        print("  Raw center: (\(centerX), \(centerY))")
        print("  Raw dimensions: \(width) x \(height)")

        // Model's expected input dimensions in pixels (square)
        let modelDimension: CGFloat = 640.0

        // Use single scale factor to preserve aspect ratio
        // Use max to match Vision's .scaleFit behavior which scales based on the smaller dimension
        let scale = max(screenWidth, screenHeight) / modelDimension
        let scaledImageSize = modelDimension * scale

        print("  Screen dimensions (points): \(screenWidth) x \(screenHeight)")
        print("  Scale factor: \(scale)")
        
        // Scale all dimensions
        let scaledWidth = CGFloat(width) * scale
        let scaledHeight = CGFloat(height) * scale
        let scaledCenterX = CGFloat(centerX) * scale
        let scaledCenterY = CGFloat(centerY) * scale
        
        // Convert from center coordinates to top-left origin
        let scaledX = scaledCenterX - (scaledWidth / 2)
        let scaledY = scaledCenterY - (scaledHeight / 2)

        let horizontalOffset = abs(screenWidth - scaledImageSize) / 2.0
        let verticalOffset = abs(screenHeight - scaledImageSize) / 2.0

        print("  Horizontal Offset: \(horizontalOffset)  ScreenWidth: \(screenWidth)  scaledImageSize: \(scaledImageSize)")
        print("  Vertical Offset: \(verticalOffset)  ScreenWidth: \(screenHeight)  scaledImageSize: \(scaledImageSize)")
        
        print("  Scaled center: (\(scaledCenterX), \(scaledCenterY))")
        print("  Scaled dimensions: \(scaledWidth) x \(scaledHeight)")
        print("  Scaled top-left: (\(scaledX), \(scaledY))")
        
        let boundingBox = CGRect(
            x: scaledX - horizontalOffset,
            y: scaledY - verticalOffset + 50,
            width: scaledWidth,
            height: scaledHeight
        )

        let detection = Detection(
            label: getClassName(for: maxClassIndex),
            confidence: maxScore,
            boundingBox: boundingBox
        )
        
        print("  ✅ Detection: \(detection.label) (\(String(format: "%.1f", maxScore * 100))%)")
        return (detection, maxScore)
    }
    
    private func findBestClass(box: Int, multiArray: MLMultiArray, numClasses: Int) -> (Float, Int) {
        var maxScore: Float = 0
        var maxClassIndex = 0
        
        // Find max score directly from class scores (first 4 values are box coordinates)
        for classIndex in 0..<numClasses {
            let score = Float(truncating: multiArray[[0, 4 + classIndex, box] as [NSNumber]])
            if score > maxScore {
                maxScore = score
                maxClassIndex = classIndex
            }
        }
        
        return (maxScore, maxClassIndex)
    }
    
    private func calculateIOU(_ rect1: CGRect, _ rect2: CGRect) -> Float {
        // Both rects are already in normalized coordinates
        let intersection = rect1.intersection(rect2)
        guard !intersection.isNull else { return 0 }
        
        let intersectionArea = Float(intersection.width * intersection.height)
        let area1 = Float(rect1.width * rect1.height)
        let area2 = Float(rect2.width * rect2.height)
        let unionArea = area1 + area2 - intersectionArea
        
        guard unionArea > 0 else { return 0 }
        return intersectionArea / unionArea
    }
    
    private func getClassName(for index: Int) -> String {
        let cocoClasses = [
            "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
            "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
            "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
            "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
            "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
            "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
            "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
            "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
            "hair drier", "toothbrush"
        ]
        
        return index < cocoClasses.count ? cocoClasses[index] : "unknown"
    }
}

extension CGRect {
    var area: CGFloat {
        return width * height
    }
} 

