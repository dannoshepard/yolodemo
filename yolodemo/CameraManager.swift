import AVFoundation
import UIKit

class CameraManager: NSObject, ObservableObject {
    @Published var previewLayer: AVCaptureVideoPreviewLayer?
    
    private let session = AVCaptureSession()
    private let videoOutput = AVCaptureVideoDataOutput()
    var onFrame: ((CVPixelBuffer) -> Void)?
    
    override init() {
        super.init()
        print("CameraManager initialized")
    }
    
    func checkPermissions() {
        let status = AVCaptureDevice.authorizationStatus(for: .video)
        print("Camera permission status: \(status.rawValue)")
        
        switch status {
        case .authorized:
            print("Camera access already authorized")
            setupAndStartSession()
        case .notDetermined:
            print("Requesting camera permission")
            AVCaptureDevice.requestAccess(for: .video) { [weak self] granted in
                print("Camera permission response: \(granted)")
                if granted {
                    DispatchQueue.main.async {
                        print("Camera permission granted, setting up session")
                        self?.setupAndStartSession()
                    }
                } else {
                    print("Camera permission denied")
                }
            }
        case .denied:
            print("Camera access denied")
        case .restricted:
            print("Camera access restricted")
        @unknown default:
            print("Unknown camera permission status")
        }
    }
    
    private func setupAndStartSession() {
        print("Beginning camera session configuration")
        session.beginConfiguration()
        
        guard let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back) else {
            print("Failed to get camera device")
            return
        }
        
        do {
            let input = try AVCaptureDeviceInput(device: device)
            if session.canAddInput(input) {
                session.addInput(input)
                print("Added camera input to session")
                
                // Configure video output for full resolution
                videoOutput.videoSettings = [
                    kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA
                ]
                
                // Set preview layer to fill
                previewLayer?.videoGravity = .resizeAspectFill
            } else {
                print("Could not add camera input to session")
            }
            
            if session.canAddOutput(videoOutput) {
                session.addOutput(videoOutput)
                videoOutput.setSampleBufferDelegate(self, queue: DispatchQueue(label: "videoQueue"))
                print("Added video output to session")
            } else {
                print("Could not add video output to session")
            }
            
            session.commitConfiguration()
            print("Committed session configuration")
            
            DispatchQueue.main.async { [weak self] in
                guard let self = self else { return }
                print("Creating preview layer")
                let previewLayer = AVCaptureVideoPreviewLayer(session: self.session)
                
                // Set initial frame (will be updated by SwiftUI view)
                previewLayer.videoGravity = .resizeAspectFill
                previewLayer.connection?.videoOrientation = .portrait
                
                self.previewLayer = previewLayer
                
                DispatchQueue.global(qos: .userInitiated).async {
                    print("Starting capture session")
                    if !self.session.isRunning {
                        self.session.startRunning()
                        print("Capture session started")
                    } else {
                        print("Session was already running")
                    }
                }
            }
        } catch {
            print("Camera setup error: \(error.localizedDescription)")
            session.commitConfiguration()
        }
    }
}

extension CameraManager: AVCaptureVideoDataOutputSampleBufferDelegate {
    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        DispatchQueue.main.async {
            self.onFrame?(pixelBuffer)
        }
    }
} 