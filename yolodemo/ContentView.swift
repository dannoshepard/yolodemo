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
        
        // Set the preview layer to fill the entire view
        previewLayer.frame = view.bounds
        previewLayer.videoGravity = .resizeAspectFill
        view.layer.addSublayer(previewLayer)
        
        return view
    }
    
    func updateUIView(_ uiView: UIView, context: Context) {
        DispatchQueue.main.async {
            // Update the preview layer to match the view's full bounds
            previewLayer.frame = uiView.bounds
        }
    }
}


struct DetectionBoxesView: View {
    let detections: [Detection]
    let geometry: GeometryProxy
    
    var body: some View {
        ZStack {
            ForEach(detections) { detection in
                Rectangle()
                    .path(in: detection.boundingBox)
                    .stroke(Color.red, lineWidth: 2)
                
                Text("\(detection.label) \(Int(detection.confidence * 100))%")
                    .foregroundColor(.white)
                    .background(Color.black.opacity(0.5))
                    .padding(4)
                    .position(
                        x: detection.boundingBox.midX,
                        y: detection.boundingBox.minY - 10
                    )
            }
        }
        .onAppear {
            // Update YOLOProcessor with current screen dimensions
            YOLOProcessor.shared.updateScreenDimensions(
                width: geometry.size.width,
                height: geometry.size.height
            )
        }
        .onChange(of: geometry.size) { newSize in
            // Update when screen size changes (e.g. rotation)
            YOLOProcessor.shared.updateScreenDimensions(
                width: newSize.width,
                height: newSize.height
            )
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
