import Vision
import CoreML
import UIKit

class YOLOProcessor {
    static let shared = YOLOProcessor()
    
    private var visionModel: VNCoreMLModel?
    private var isModelLoaded = false
    
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
    
    private func softmax(_ x: [Float]) -> [Float] {
        let maxVal = x.max() ?? 0
        let exp_x = x.map { exp($0 - maxVal) }
        let sum = exp_x.reduce(0, +)
        return exp_x.map { $0 / sum }
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
        
        // Calculate center square crop region
        let cropSize = min(width, height)
        let cropX = (width - cropSize) / 2
        let cropY = (height - cropSize) / 2
        
        // Convert to normalized coordinates [0,1]
        let normalizedRect = CGRect(
            x: CGFloat(cropX) / CGFloat(width),
            y: CGFloat(cropY) / CGFloat(height),
            width: CGFloat(cropSize) / CGFloat(width),
            height: CGFloat(cropSize) / CGFloat(height)
        )
        
        print("Processing center square region: \(cropSize)x\(cropSize)")
        
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
            
            // Configure request to crop center square
            request.imageCropAndScaleOption = .scaleFit
            request.regionOfInterest = normalizedRect
            
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
        
        for box in 0..<numBoxes {
            let x = Float(truncating: multiArray[[0, 0, box] as [NSNumber]])
            let y = Float(truncating: multiArray[[0, 1, box] as [NSNumber]])
            let w = Float(truncating: multiArray[[0, 2, box] as [NSNumber]])
            let h = Float(truncating: multiArray[[0, 3, box] as [NSNumber]])
            
            // Skip invalid boxes
            guard w > 0 && h > 0 && w < 640 && h < 640 else { continue }
            
            // Find the highest class score
            var maxScore: Float = 0
            var maxClassIndex = 0
            
            for classIndex in 0..<numClasses {
                let score = Float(truncating: multiArray[[0, 4 + classIndex, box] as [NSNumber]])
                if score > maxScore {
                    maxScore = score
                    maxClassIndex = classIndex
                }
            }
            
            // Higher confidence threshold
            if maxScore > 0.35 {
                // Create normalized bounding box
                let boundingBox = CGRect(
                    x: CGFloat((y - h/2) / 640.0),  // YOLO's y is our x
                    y: CGFloat(1.0 - ((x + w/2) / 640.0)),  // YOLO's x is our y, flipped
                    width: CGFloat(h / 640.0),  // YOLO's h is our width
                    height: CGFloat(w / 640.0)   // YOLO's w is our height
                )
                
                let detection = Detection(
                    label: getClassName(for: maxClassIndex),
                    confidence: maxScore,
                    boundingBox: boundingBox
                )
                
                candidateDetections.append((detection, maxScore))
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
                return iou < 0.45
            }
        }
        
        return finalDetections
    }
    
    private func calculateIOU(_ rect1: CGRect, _ rect2: CGRect) -> Float {
        let intersectionRect = rect1.intersection(rect2)
        guard !intersectionRect.isNull else { return 0 }
        
        // Calculate areas step by step
        let area1 = Float(rect1.width * rect1.height)
        let area2 = Float(rect2.width * rect2.height)
        let intersectionArea = Float(intersectionRect.width * intersectionRect.height)
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

