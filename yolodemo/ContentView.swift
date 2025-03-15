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
    @State private var showControls = false
    
    // YOLO parameters
    @State private var confidenceThreshold: Double = Double(YOLOProcessor.shared.config.confidenceThreshold)
    @State private var iouThreshold: Double = Double(YOLOProcessor.shared.config.iouThreshold)
    @State private var minBoxSize: Double = Double(YOLOProcessor.shared.config.minBoxSize)
    @State private var maxBoxSize: Double = Double(YOLOProcessor.shared.config.maxBoxSize)
    
    var body: some View {
        GeometryReader { geometry in
            ZStack {
                if let previewLayer = cameraManager.previewLayer {
                    CameraPreviewView(previewLayer: previewLayer)
                        .ignoresSafeArea()
                    
                    DetectionBoxesView(detections: detections, geometry: geometry)
                        .ignoresSafeArea()
                    
                    VStack {
                        // Debug overlay
                        HStack {
                            VStack(alignment: .leading) {
                                Text("Processing: \(isProcessingFrame ? "Yes" : "No")")
                                    .foregroundColor(.green)
                                Text("Detections: \(detections.count)")
                                    .foregroundColor(.green)
                            }
                            .padding()
                            .background(Color.black.opacity(0.5))
                            
                            Spacer()
                            
                            Button(action: { showControls.toggle() }) {
                                Image(systemName: "slider.horizontal.3")
                                    .foregroundColor(.white)
                                    .padding()
                                    .background(Color.black.opacity(0.5))
                            }
                        }
                        .padding()
                        
                        Spacer()
                        
                        // Parameter controls
                        if showControls {
                            VStack(spacing: 10) {
                                ParameterSlider(value: $confidenceThreshold,
                                              range: 0.1...1.0,
                                              title: "Confidence",
                                              format: "%.2f") { newValue in
                                    YOLOProcessor.shared.config.confidenceThreshold = Float(newValue)
                                }
                                
                                ParameterSlider(value: $iouThreshold,
                                              range: 0.1...1.0,
                                              title: "IoU",
                                              format: "%.2f") { newValue in
                                    YOLOProcessor.shared.config.iouThreshold = Float(newValue)
                                }
                                
                                ParameterSlider(value: $minBoxSize,
                                              range: 5...100,
                                              title: "Min Size",
                                              format: "%.0f") { newValue in
                                    YOLOProcessor.shared.config.minBoxSize = Float(newValue)
                                }
                                
                                ParameterSlider(value: $maxBoxSize,
                                              range: 100...640,
                                              title: "Max Size",
                                              format: "%.0f") { newValue in
                                    YOLOProcessor.shared.config.maxBoxSize = Float(newValue)
                                }
                            }
                            .padding()
                            .background(Color.black.opacity(0.7))
                        }
                    }
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
            await MainActor.run {
                self.detections = newDetections
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

struct ParameterSlider: View {
    @Binding var value: Double
    let range: ClosedRange<Double>
    let title: String
    let format: String
    let onChanged: (Double) -> Void
    
    var body: some View {
        VStack(alignment: .leading) {
            Text("\(title): \(String(format: format, value))")
                .foregroundColor(.white)
            Slider(value: $value, in: range) { _ in
                onChanged(value)
            }
        }
    }
}

#Preview {
    ContentView()
}
