//
//  ContentView.swift
//  yolodemo
//
//  Created by Danno on 3/15/25.
//

import SwiftUI
import AVFoundation
import CoreML
import Vision

struct ContentView: View {
    @StateObject private var cameraManager = CameraManager()
    @State private var detections: [Detection] = []
    @State private var lastProcessedTime = Date()
    @State private var isProcessingFrame = false
    
    var body: some View {
        GeometryReader { geometry in
            ZStack {
                if let previewLayer = cameraManager.previewLayer {
                    CameraPreviewView(previewLayer: previewLayer)
                        .ignoresSafeArea()
                        .frame(width: geometry.size.width, height: geometry.size.height)
                    
                    DetectionBoxesView(detections: detections, geometry: geometry)
                        .ignoresSafeArea()
                    
                    // Debug overlay
                    VStack {
                        Text("Processing: \(isProcessingFrame ? "Yes" : "No")")
                            .foregroundColor(.green)
                            .padding()
                            .background(Color.black.opacity(0.5))
                        Text("Detections: \(detections.count)")
                            .foregroundColor(.green)
                            .padding()
                            .background(Color.black.opacity(0.5))
                    }
                    .position(x: 100, y: 50)
                } else {
                    Color.black
                        .ignoresSafeArea()
                    Text("Camera not available")
                        .foregroundColor(.white)
                }
            }
        }
        .onAppear {
            cameraManager.checkPermissions()
            cameraManager.onFrame = { pixelBuffer in
                // Throttle frame processing to every 100ms
                let now = Date()
                guard now.timeIntervalSince(lastProcessedTime) >= 0.1 else { return }
                lastProcessedTime = now
                
                Task {
                    await processFrame(pixelBuffer)
                }
            }
        }
    }
    
    private func processFrame(_ pixelBuffer: CVPixelBuffer) async {
        guard !isProcessingFrame else { return }
        
        isProcessingFrame = true
        defer { isProcessingFrame = false }
        
        do {
            let newDetections = try await YOLOProcessor.shared.detect(in: pixelBuffer)
            print("Raw detections count: \(newDetections.count)")
            
            await MainActor.run {
                // Lower the confidence threshold for testing
                self.detections = newDetections.filter { detection in
                    let passed = detection.confidence > 0.2 // Lower threshold for testing
                    print("Detection \(detection.label): confidence \(detection.confidence) passed filter: \(passed)")
                    return passed
                }
                print("Filtered detections count: \(self.detections.count)")
            }
        } catch {
            print("Detection error: \(error)")
        }
    }
}

struct CameraPreviewView: UIViewRepresentable {
    let previewLayer: AVCaptureVideoPreviewLayer
    
    func makeUIView(context: Context) -> UIView {
        let view = UIView(frame: UIScreen.main.bounds)
        
        // Calculate square dimensions
        let squareSize = min(view.bounds.width, view.bounds.height)
        let x = (view.bounds.width - squareSize) / 2
        let y = (view.bounds.height - squareSize) / 2
        
        // Set square frame
        previewLayer.frame = CGRect(x: x, y: y, width: squareSize, height: squareSize)
        previewLayer.videoGravity = .resizeAspectFill
        view.layer.addSublayer(previewLayer)
        return view
    }
    
    func updateUIView(_ uiView: UIView, context: Context) {
        DispatchQueue.main.async {
            // Maintain square aspect ratio when view updates
            let squareSize = min(uiView.bounds.width, uiView.bounds.height)
            let x = (uiView.bounds.width - squareSize) / 2
            let y = (uiView.bounds.height - squareSize) / 2
            previewLayer.frame = CGRect(x: x, y: y, width: squareSize, height: squareSize)
        }
    }
}

struct DetectionBoxesView: View {
    let detections: [Detection]
    let geometry: GeometryProxy
    
    var body: some View {
        let squareSize = min(geometry.size.width, geometry.size.height)
        let x = (geometry.size.width - squareSize) / 2
        let y = (geometry.size.height - squareSize) / 2
        
        ZStack {
            ForEach(detections) { detection in
                let rect = detection.boundingBox
                Rectangle()
                    .path(in: CGRect(
                        x: x + (rect.minX * squareSize),
                        y: y + (rect.minY * squareSize),
                        width: rect.width * squareSize,
                        height: rect.height * squareSize
                    ))
                    .stroke(Color.red, lineWidth: 2)
                
                Text("\(detection.label) \(Int(detection.confidence * 100))%")
                    .foregroundColor(.white)
                    .background(Color.black.opacity(0.5))
                    .padding(4)
                    .position(
                        x: x + (rect.minX * squareSize),
                        y: y + (rect.minY * squareSize) - 10
                    )
            }
        }
    }
}

struct Detection: Identifiable {
    let id = UUID()
    let label: String
    let confidence: Float
    let boundingBox: CGRect
}

#Preview {
    ContentView()
}
