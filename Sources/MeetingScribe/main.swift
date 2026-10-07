import AppKit
import AVFoundation
import AudioToolbox
import CoreGraphics
import CoreMedia
import Darwin
import ScreenCaptureKit
import SwiftUI
import UniformTypeIdentifiers

enum Speaker: String, CaseIterable {
    case selfUser = "自分"
    case others = "相手"
}

private struct TranscriptLine: Identifiable {
    let id = UUID()
    let offset: TimeInterval
    let endOffset: TimeInterval?
    let speaker: Speaker
    let text: String

    var timeLabel: String {
        let seconds = max(0, Int(offset))
        return String(format: "%02d:%02d:%02d", seconds / 3600, (seconds / 60) % 60, seconds % 60)
    }
}

private struct AudioPathMetrics {
    var callbacks = 0
    var inputFrames = 0
    var sampleRate = 0.0
    var channels = 0
    var convertedFrames = 0
    var sumSquares = 0.0
    var sampleCount = 0
    var peak = Float.zero
    var segments = 0
}

private struct STTMetrics {
    var queued = 0
    var succeeded = 0
    var empty = 0
    var failed = 0
}

@MainActor
private final class AppModel: ObservableObject {
    @Published var isRecording = false
    @Published var isProcessing = false
    @Published var isStarting = false
    @Published var isStopping = false
    @Published var isAwaitingBatchChoice = false
    @Published var isBatchTranscribing = false
    @Published var batchProgress = 0.0
    @Published var batchProgressDetail = ""
    @Published var estimatedBatchCompletion: Date?
    @Published var lastBatchProgressAt: Date?
    @Published var status = "モデルと whisper-cli を準備してください"
    @Published var lines: [TranscriptLine] = []
    @Published var modelPath: URL? = AppModel.defaultModelURL
    @Published var batchModelPath: URL? = AppModel.defaultBatchModelURL
    @Published var whisperPath: URL? = AppModel.findWhisperCLI()
    @Published var downloadProgress: Double?
    @Published var batchDownloadProgress: Double?
    @Published var meetingPrompt: String?
    @Published var autoDetect = true
    @Published var audioDiagnosticSummary = ""

    private var capture: MeetingAudioCapture?
    private var audioArchive: SessionAudioArchive?
    private var archiveFailureMessage: String?
    private var batchFailureMessage: String?
    private var batchStarted = false
    private var batchDecisionMade = false
    private var batchRunID = UUID()
    private var batchFallbackSpeakers = Set<Speaker>()
    private var batchProgressStartedAt: Date?
    private var lastObservedBatchFraction = 0.0
    private var lastObservedBatchProgressAt: Date?
    private var smoothedBatchSecondsPerFraction: TimeInterval?
    private let inferenceQueue = OperationQueue()
    private var sessionStartedAt = Date()
    private var sessionStartedUptime: TimeInterval?
    private var detectorTask: Task<Void, Never>?
    private var pendingSegments = 0
    private var failedSegments = 0
    private var droppedAudioSeconds: TimeInterval = 0
    private var captureFailureMessage: String?
    private var queuedAudioSeconds: TimeInterval = 0
    private let maximumQueuedAudioSeconds: TimeInterval = 180
    private var downloadProgressObservation: NSKeyValueObservation?
    private var batchDownloadProgressObservation: NSKeyValueObservation?
    private var backendValidated = false
    private var captureStartFailureMessage: String?
    private var meetingWasVisible = false
    private var dismissedCandidate = false
    private var dismissedEnd = false
    private var sttMetrics: [Speaker: STTMetrics] = [.selfUser: STTMetrics(), .others: STTMetrics()]

    init() {
        Task.detached(priority: .utility) { SessionAudioArchive.removeAbandonedArchives() }
        inferenceQueue.maxConcurrentOperationCount = 1
        inferenceQueue.qualityOfService = .userInitiated
        if ready {
            status = batchModelPath == nil
                ? "リアルタイム認識は準備済みです。録音後用のWhisper large-v3を取得してください。"
                : "準備完了です。開始を押してください。"
        }
        startMeetingMonitor()
    }

    static let modelFile = "ggml-kotoba-whisper-v2.0-q5_0.bin"
    nonisolated static let modelSizeBytes: Int64 = 537_819_875
    static let modelURL = URL(string: "https://huggingface.co/kotoba-tech/kotoba-whisper-v2.0-ggml/resolve/a10e12364e78988c774a6a60a83a6f65ffd60c01/ggml-kotoba-whisper-v2.0-q5_0.bin")!
    static let batchModelFile = "ggml-large-v3.bin"
    static let batchModelURL = URL(string: "https://huggingface.co/ggerganov/whisper.cpp/resolve/c521a4b02f422512d734391fdf08bb08c0862f68/ggml-large-v3.bin")!
    static var defaultModelURL: URL? {
        let url = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("MeetingScribe/Models/\(modelFile)")
        return isValidModel(url) ? url : nil
    }

    static var defaultBatchModelURL: URL? {
        let url = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("MeetingScribe/Models/\(batchModelFile)")
        return isValidBatchModel(url) ? url : nil
    }

    static func isValidModel(_ url: URL) -> Bool {
        guard let attributes = try? FileManager.default.attributesOfItem(atPath: url.path),
              let size = attributes[.size] as? NSNumber else { return false }
        return size.int64Value == modelSizeBytes
    }

    nonisolated static func isValidBatchModel(_ url: URL) -> Bool {
        guard let attributes = try? FileManager.default.attributesOfItem(atPath: url.path),
              let size = attributes[.size] as? NSNumber,
              (3_000_000_000...3_400_000_000).contains(size.int64Value),
              let handle = try? FileHandle(forReadingFrom: url),
              let magic = try? handle.read(upToCount: 4),
              magic == Data([0x6c, 0x6d, 0x67, 0x67]) else { return false }
        try? handle.close()
        return true
    }

    static func findWhisperCLI() -> URL? {
        let bundledCLI = Bundle.main.bundleURL
            .appendingPathComponent("Contents/Frameworks/whisper-cli")
        let savedPath = UserDefaults.standard.string(forKey: "whisperCLIPath")
        let candidates = [bundledCLI.path, savedPath, "/opt/homebrew/bin/whisper-cli", "/usr/local/bin/whisper-cli"].compactMap { $0 }
        return candidates.map(URL.init(fileURLWithPath:)).first { FileManager.default.isExecutableFile(atPath: $0.path) }
    }

    var ready: Bool { modelPath.map(Self.isValidModel) == true && whisperPath.map { FileManager.default.isExecutableFile(atPath: $0.path) && $0.lastPathComponent == "whisper-cli" } == true }

    func downloadModel() {
        guard downloadProgress == nil, batchDownloadProgress == nil else { return }
        let destination = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("MeetingScribe/Models/\(Self.modelFile)")
        do { try FileManager.default.createDirectory(at: destination.deletingLastPathComponent(), withIntermediateDirectories: true) }
        catch { status = "モデル保存先を作成できません: \(error.localizedDescription)"; return }
        downloadProgress = 0
        status = "日本語モデルをダウンロード中（約538 MB）"
        let task = URLSession.shared.downloadTask(with: Self.modelURL) { [weak self] temporaryURL, response, error in
            // URLSession deletes its temporary download file when this callback returns.
            // Move it here, before hopping to the main actor to update the UI.
            let result: Result<Void, Error>
            if let error { result = .failure(error) }
            else if let temporaryURL, let response = response as? HTTPURLResponse, (200..<300).contains(response.statusCode) {
                do {
                    let fileSize = (try temporaryURL.resourceValues(forKeys: [.fileSizeKey])).fileSize ?? 0
                    guard Int64(fileSize) == Self.modelSizeBytes else { throw CaptureError.download }
                    if FileManager.default.fileExists(atPath: destination.path) { try FileManager.default.removeItem(at: destination) }
                    try FileManager.default.moveItem(at: temporaryURL, to: destination)
                    result = .success(())
                } catch { result = .failure(error) }
            } else { result = .failure(CaptureError.download) }
            Task { @MainActor in
                guard let self else { return }
                defer { self.downloadProgress = nil }
                self.downloadProgressObservation = nil
                switch result {
                case .success:
                    self.modelPath = destination
                    self.backendValidated = false
                    self.status = "モデルを準備しました。会議音声は端末内で処理されます。"
                case .failure(let error):
                    self.status = "モデル取得に失敗しました: \(error.localizedDescription)"
                }
            }
        }
        downloadProgressObservation = task.progress.observe(\.fractionCompleted) { [weak self] progress, _ in
            Task { @MainActor in self?.downloadProgress = progress.fractionCompleted }
        }
        task.resume()
    }

    func downloadBatchModel() {
        guard batchDownloadProgress == nil, downloadProgress == nil, !isRecording, !isProcessing, !isStarting, !isStopping else { return }
        let destination = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("MeetingScribe/Models/\(Self.batchModelFile)")
        do { try FileManager.default.createDirectory(at: destination.deletingLastPathComponent(), withIntermediateDirectories: true) }
        catch { status = "モデル保存先を作成できません: \(error.localizedDescription)"; return }
        batchDownloadProgress = 0
        status = "Whisper large-v3 をダウンロード中（約3.1 GB）"
        let task = URLSession.shared.downloadTask(with: Self.batchModelURL) { [weak self] temporaryURL, response, error in
            let result: Result<Void, Error>
            if let error { result = .failure(error) }
            else if let temporaryURL, let response = response as? HTTPURLResponse, (200..<300).contains(response.statusCode) {
                do {
                    guard Self.isValidBatchModel(temporaryURL) else { throw CaptureError.download }
                    if FileManager.default.fileExists(atPath: destination.path) { try FileManager.default.removeItem(at: destination) }
                    try FileManager.default.moveItem(at: temporaryURL, to: destination)
                    result = .success(())
                } catch { result = .failure(error) }
            } else { result = .failure(CaptureError.download) }
            Task { @MainActor in
                guard let self else { return }
                defer {
                    self.batchDownloadProgress = nil
                    self.batchDownloadProgressObservation = nil
                }
                switch result {
                case .success:
                    self.batchModelPath = destination
                    self.status = "Whisper large-v3 を準備しました。録音後に再処理します。"
                case .failure(let error):
                    self.status = "高精度モデルの取得に失敗しました: \(error.localizedDescription)"
                }
            }
        }
        batchDownloadProgressObservation = task.progress.observe(\.fractionCompleted) { [weak self] progress, _ in
            Task { @MainActor in self?.batchDownloadProgress = progress.fractionCompleted }
        }
        task.resume()
    }

    func chooseWhisperCLI() {
        let panel = NSOpenPanel()
        panel.title = "whisper-cli を選択"
        panel.message = "ローカルでビルドした whisper.cpp の whisper-cli を選択してください。"
        panel.canChooseDirectories = false
        panel.canChooseFiles = true
        panel.allowsMultipleSelection = false
        if panel.runModal() == .OK, let url = panel.url {
            guard url.lastPathComponent == "whisper-cli", FileManager.default.isExecutableFile(atPath: url.path) else {
                status = "whisper.cpp の whisper-cli 実行ファイルを選択してください。"; return
            }
            whisperPath = url
            UserDefaults.standard.set(url.path, forKey: "whisperCLIPath")
            backendValidated = false
            status = ready ? "準備完了です。開始を押してください。" : "Kotoba-Whisper モデルを準備してください。"
        }
    }

    func start() {
        guard ready, batchDownloadProgress == nil, !isRecording, !isProcessing, !isAwaitingBatchChoice, !isStarting, !isStopping else { return }
        isStarting = true
        Task {
            do {
                captureStartFailureMessage = nil
                if !backendValidated, let modelPath, let whisperPath {
                    status = "モデルと whisper.cpp を確認しています…"
                    try await Task.detached(priority: .userInitiated) {
                        try AppModel.validateBackend(modelPath: modelPath, whisperPath: whisperPath)
                    }.value
                    backendValidated = true
                }
                let microphoneAuthorization = AVCaptureDevice.authorizationStatus(for: .audio)
                if microphoneAuthorization == .notDetermined {
                    guard await AVCaptureDevice.requestAccess(for: .audio) else { throw CaptureError.microphonePermission }
                } else if microphoneAuthorization != .authorized {
                    throw CaptureError.microphonePermission
                }
                if !CGPreflightScreenCaptureAccess() {
                    let granted = CGRequestScreenCaptureAccess()
                    guard granted, CGPreflightScreenCaptureAccess() else {
                        throw CaptureError.screenCapturePermission
                    }
                }
                let content = try await SCShareableContent.excludingDesktopWindows(false, onScreenWindowsOnly: true)
                guard let display = content.displays.first else { throw CaptureError.noDisplay }
                sessionStartedAt = Date()
                lines = []
                pendingSegments = 0
                failedSegments = 0
                droppedAudioSeconds = 0
                sttMetrics = [.selfUser: STTMetrics(), .others: STTMetrics()]
                audioDiagnosticSummary = ""
                captureFailureMessage = nil
                queuedAudioSeconds = 0
                archiveFailureMessage = nil
                batchFailureMessage = nil
                batchStarted = false
                batchDecisionMade = false
                batchRunID = UUID()
                isAwaitingBatchChoice = false
                isBatchTranscribing = false
                batchProgress = 0
                batchProgressDetail = ""
                estimatedBatchCompletion = nil
                lastBatchProgressAt = nil
                batchProgressStartedAt = nil
                lastObservedBatchFraction = 0
                lastObservedBatchProgressAt = nil
                smoothedBatchSecondsPerFraction = nil
                batchFallbackSpeakers = []
                let archive = try SessionAudioArchive()
                audioArchive = archive
                let hostClockStart = CMTimeGetSeconds(CMClockGetTime(CMClockGetHostTimeClock()))
                let handler = MeetingAudioCapture(sessionStart: hostClockStart, emit: { [weak self] speaker, samples, offset in
                    self?.enqueue(samples: samples, speaker: speaker, offset: offset)
                }, onError: { [weak self] message in
                    self?.status = message
                    self?.captureFailureMessage = message
                }, onStreamFailure: { [weak self] message in
                    guard let self else { return }
                    self.captureStartFailureMessage = message
                    if self.isRecording { self.captureFailed(message) }
                }, archive: archive)
                let config = SCStreamConfiguration()
                config.width = 2
                config.height = 2
                config.minimumFrameInterval = CMTime(value: 1, timescale: 2)
                config.queueDepth = 3
                config.capturesAudio = true
                config.captureMicrophone = true
                config.excludesCurrentProcessAudio = true
                config.sampleRate = 16_000
                config.channelCount = 1
                let ownApp = content.applications.first { $0.bundleIdentifier == Bundle.main.bundleIdentifier }
                let filter = SCContentFilter(display: display, excludingApplications: ownApp.map { [$0] } ?? [], exceptingWindows: [])
                let stream = SCStream(filter: filter, configuration: config, delegate: handler)
                try stream.addStreamOutput(handler, type: .audio, sampleHandlerQueue: handler.callbackQueue)
                try stream.addStreamOutput(handler, type: .microphone, sampleHandlerQueue: handler.callbackQueue)
                self.capture = handler
                try await stream.startCapture()
                if let captureStartFailureMessage { throw CaptureError.inference(captureStartFailureMessage) }
                handler.stream = stream
                sessionStartedUptime = ProcessInfo.processInfo.systemUptime
                isStarting = false
                isRecording = true
                status = "文字起こし中（音声はこの Mac 内で処理）"
                dismissedCandidate = false
                dismissedEnd = false
            } catch {
                if let archive = audioArchive {
                    Task.detached(priority: .utility) {
                        _ = await archive.finish()
                        archive.remove()
                    }
                }
                audioArchive = nil
                self.capture = nil
                isStarting = false
                status = "録音を開始できません: \(error.localizedDescription)。画面収録とマイクの権限を確認してください。"
            }
        }
    }

    func stop() {
        guard isRecording, !isStopping, let capture else { return }
        let elapsed = max(0, ProcessInfo.processInfo.systemUptime - (sessionStartedUptime ?? ProcessInfo.processInfo.systemUptime))
        isStopping = true
        status = "録音を停止しています…"
        Task {
            do { try await capture.stop() }
            catch {
                isStopping = false
                status = "録音を停止できませんでした。キャプチャが継続している可能性があります: \(error.localizedDescription)"
                return
            }
            await capture.flushAfterPendingCallbacks()
            audioDiagnosticSummary = String(format: "録音経過 %.1f秒\n%@", elapsed, capture.diagnosticSummary())
            archiveFailureMessage = await capture.finishArchive()
            if let archive = audioArchive { audioDiagnosticSummary += "\n保存診断: \(archive.diagnosticSummary())" }
            sessionStartedUptime = nil
            isRecording = false
            isStopping = false
            isProcessing = true
            self.capture = nil
            status = "録音を停止し、残りの音声を処理しています…"
            finishWhenDrained()
        }
    }

    private func captureFailed(_ message: String) {
        guard isRecording, !isStopping, let capture else { status = message; return }
        let elapsed = max(0, ProcessInfo.processInfo.systemUptime - (sessionStartedUptime ?? ProcessInfo.processInfo.systemUptime))
        isStopping = true
        status = message
        Task {
            await capture.flushAfterPendingCallbacks()
            audioDiagnosticSummary = String(format: "録音経過 %.1f秒\n%@", elapsed, capture.diagnosticSummary())
            archiveFailureMessage = await capture.finishArchive()
            if let archive = audioArchive { audioDiagnosticSummary += "\n保存診断: \(archive.diagnosticSummary())" }
            sessionStartedUptime = nil
            isRecording = false
            isStopping = false
            isProcessing = true
            self.capture = nil
            captureFailureMessage = message
            finishWhenDrained()
        }
    }

    private func enqueue(samples: [Float], speaker: Speaker, offset: TimeInterval) {
        guard !samples.isEmpty, let modelPath, let whisperPath else { return }
        let audioSeconds = Double(samples.count) / 16_000
        guard queuedAudioSeconds + audioSeconds <= maximumQueuedAudioSeconds else {
            droppedAudioSeconds += audioSeconds
            status = String(format: "推論が追いつかず %.0f 秒の音声を破棄しました。結果に欠落があります。", droppedAudioSeconds)
            return
        }
        queuedAudioSeconds += audioSeconds
        pendingSegments += 1
        let temp = FileManager.default.temporaryDirectory.appendingPathComponent("MeetingScribe-\(UUID().uuidString)")
        let wav = temp.appendingPathExtension("wav")
        let output = temp.appendingPathExtension("txt")
        let operation = BlockOperation { [weak self] in
            var recognizedText: String?
            var failureMessage: String?
            defer {
                try? FileManager.default.removeItem(at: wav)
                try? FileManager.default.removeItem(at: output)
                try? FileManager.default.removeItem(at: temp.appendingPathExtension("log"))
                Task { @MainActor in
                    guard let self else { return }
                    if let recognizedText, !recognizedText.isEmpty {
                        self.lines.append(TranscriptLine(offset: offset, endOffset: offset + audioSeconds, speaker: speaker, text: recognizedText))
                        self.lines.sort { $0.offset < $1.offset }
                        self.sttMetrics[speaker, default: STTMetrics()].succeeded += 1
                    } else if failureMessage == nil {
                        self.sttMetrics[speaker, default: STTMetrics()].empty += 1
                    }
                    if let failureMessage {
                        self.failedSegments += 1
                        self.sttMetrics[speaker, default: STTMetrics()].failed += 1
                        self.status = "文字起こしに失敗した区間があります（既存の結果は保持）: \(failureMessage)"
                    }
                    self.pendingSegments = max(0, self.pendingSegments - 1)
                    self.queuedAudioSeconds = max(0, self.queuedAudioSeconds - audioSeconds)
                    self.finishWhenDrained()
                }
            }
            do {
                try Self.writeWAV(samples, to: wav)
                let process = Process()
                process.executableURL = whisperPath
                process.arguments = ["-m", modelPath.path, "-l", "ja", "-f", wav.path, "-t", "4", "-nt", "-np", "-otxt", "-of", temp.path]
                let errorLog = temp.appendingPathExtension("log")
                FileManager.default.createFile(atPath: errorLog.path, contents: nil)
                let outputHandle = try FileHandle(forWritingTo: errorLog)
                process.standardError = outputHandle
                process.standardOutput = outputHandle
                let exited = DispatchSemaphore(value: 0)
                process.terminationHandler = { _ in exited.signal() }
                try process.run()
                if exited.wait(timeout: .now() + 180) == .timedOut {
                    if process.isRunning { process.terminate() }
                    if exited.wait(timeout: .now() + 5) == .timedOut {
                        _ = Darwin.kill(process.processIdentifier, SIGKILL)
                        _ = exited.wait(timeout: .now() + 5)
                    }
                    outputHandle.closeFile()
                    throw CaptureError.inference("推論が3分でタイムアウトしました。")
                }
                process.waitUntilExit()
                outputHandle.closeFile()
                guard process.terminationStatus == 0 else {
                    let detail = (try? String(contentsOf: errorLog, encoding: .utf8)) ?? ""
                    throw CaptureError.inference(detail.trimmingCharacters(in: .whitespacesAndNewlines))
                }
                let text = try String(contentsOf: output, encoding: .utf8).trimmingCharacters(in: .whitespacesAndNewlines)
                recognizedText = text
            } catch {
                failureMessage = error.localizedDescription
            }
        }
        sttMetrics[speaker, default: STTMetrics()].queued += 1
        inferenceQueue.addOperation(operation)
    }

    private func finishWhenDrained() {
        guard !isRecording, !isStopping, isProcessing, pendingSegments == 0 else { return }
        guard !batchStarted else { return }
        batchStarted = true
        isProcessing = false
        isAwaitingBatchChoice = true
        status = "録音を終了しました。録音後の処理を選択してください。"
    }

    func chooseBatchTranscription() {
        guard !batchDecisionMade else { return }
        batchDecisionMade = true
        isAwaitingBatchChoice = false
        isProcessing = true
        isBatchTranscribing = true
        batchProgress = 0
        batchProgressDetail = "whisper-cliを起動しています…"
        estimatedBatchCompletion = nil
        lastBatchProgressAt = Date()
        batchProgressStartedAt = Date()
        lastObservedBatchFraction = 0
        lastObservedBatchProgressAt = nil
        smoothedBatchSecondsPerFraction = nil
        beginBatchTranscription()
    }

    func chooseRealtimeOnly() {
        guard !batchDecisionMade else { return }
        batchDecisionMade = true
        isAwaitingBatchChoice = false
        completeTranscription(outcome: nil, wasSkipped: true)
    }

    private func beginBatchTranscription() {
        guard let archive = audioArchive,
              let modelPath = batchModelPath,
              let whisperPath,
              Self.isValidBatchModel(modelPath),
              archiveFailureMessage == nil else {
            batchFailureMessage = archiveFailureMessage ?? "高精度モデルまたはバッチ用音声が未準備です。リアルタイム結果を保持します。"
            completeTranscription(outcome: nil)
            return
        }
        status = "録音音声を再処理しています…"
        let archiveDirectory = archive.directory
        let capturedSpeakers = archive.speakersWithInput
        let runID = batchRunID
        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            let outcome = BatchTranscriber.transcribe(
                archiveDirectory: archiveDirectory,
                capturedSpeakers: capturedSpeakers,
                modelURL: modelPath,
                whisperURL: whisperPath,
                progress: { [weak self] update in
                    Task { @MainActor [weak self] in self?.updateBatchProgress(update, runID: runID) }
                }
            )
            Task { @MainActor in
                guard let self, self.batchRunID == runID else { return }
                self.completeTranscription(outcome: outcome)
            }
        }
    }

    private func updateBatchProgress(_ update: BatchProgressUpdate, runID: UUID) {
        guard isBatchTranscribing, batchRunID == runID else { return }
        let now = Date()
        let fraction = min(1, max(batchProgress, update.fractionCompleted))
        batchProgress = fraction
        lastBatchProgressAt = now
        if let speaker = update.speaker, update.totalChunks > 0 {
            batchProgressDetail = String(
                format: "%@ 音声 %d/%d（この区間 %.0f%%）",
                speaker.rawValue,
                update.chunkNumber,
                update.totalChunks,
                update.chunkFraction * 100
            )
        } else {
            batchProgressDetail = "後処理を完了しています…"
        }

        guard fraction > lastObservedBatchFraction,
              let startedAt = batchProgressStartedAt else { return }
        guard let previousAt = lastObservedBatchProgressAt else {
            lastObservedBatchFraction = fraction
            lastObservedBatchProgressAt = now
            return
        }
        let delta = fraction - lastObservedBatchFraction
        let interval = now.timeIntervalSince(previousAt)
        guard delta >= 0.008, interval >= 0.5 else { return }
        let sampleSecondsPerFraction = interval / delta
        if let current = smoothedBatchSecondsPerFraction {
            let bounded = min(current * 2, max(current * 0.5, sampleSecondsPerFraction))
            smoothedBatchSecondsPerFraction = current * 0.7 + bounded * 0.3
        } else {
            smoothedBatchSecondsPerFraction = sampleSecondsPerFraction
        }
        lastObservedBatchFraction = fraction
        lastObservedBatchProgressAt = now

        let elapsed = now.timeIntervalSince(startedAt)
        if fraction >= 0.025, elapsed >= 5,
           let secondsPerFraction = smoothedBatchSecondsPerFraction {
            estimatedBatchCompletion = now.addingTimeInterval(max(2, (1 - fraction) * secondsPerFraction))
        }
    }

    func batchRemainingDescription(at date: Date) -> String {
        guard let estimatedBatchCompletion else { return "残り時間を推定しています…" }
        if let lastBatchProgressAt, date.timeIntervalSince(lastBatchProgressAt) > 90 {
            return "進捗更新を待っています。残り時間を再計算中…"
        }
        if estimatedBatchCompletion <= date {
            return "完了予定を過ぎたため、残り時間を再計算中…"
        }
        let remaining = max(0, estimatedBatchCompletion.timeIntervalSince(date))
        if remaining < 60 { return String(format: "完了まで約%d秒", max(1, Int(remaining.rounded()))) }
        if remaining < 3_600 {
            let minutes = Int(remaining) / 60
            let seconds = Int(remaining) % 60
            return seconds < 15 ? "完了まで約\(minutes)分" : String(format: "完了まで約%d分%d秒", minutes, seconds)
        }
        let hours = Int(remaining) / 3_600
        let minutes = (Int(remaining) % 3_600) / 60
        return "完了まで約\(hours)時間\(minutes)分"
    }

    private func completeTranscription(outcome: BatchTranscriptionOutcome?, wasSkipped: Bool = false) {
        if let outcome {
            batchFallbackSpeakers = Set(outcome.failuresBySpeaker.keys)
            var finalLines = lines.filter { outcome.failuresBySpeaker[$0.speaker] != nil }
            var coverageWarnings: [String] = []
            for speaker in Speaker.allCases {
                guard let segments = outcome.segmentsBySpeaker[speaker] else { continue }
                let realtimeLines = lines.filter { $0.speaker == speaker }
                if let qualityWarning = BatchTranscriptQualityGuard.fallbackReason(
                    realtimeTexts: realtimeLines.map(\.text),
                    batchSegments: segments
                ) {
                    batchFallbackSpeakers.insert(speaker)
                    finalLines.append(contentsOf: realtimeLines)
                    coverageWarnings.append("\(speaker.rawValue): \(qualityWarning)")
                    continue
                }
                let ranges = (outcome.audioRangesBySpeaker[speaker] ?? []).sorted { $0.start < $1.start }
                let coverageComplete = realtimeLines.allSatisfy { line in
                    let utteranceEnd = line.endOffset ?? line.offset
                    var coveredUntil = line.offset
                    for range in ranges where range.end >= coveredUntil - 0.02 {
                        guard range.start <= coveredUntil + 0.02 else { break }
                        coveredUntil = max(coveredUntil, range.end)
                        if coveredUntil >= utteranceEnd - 0.02 { return true }
                    }
                    return coveredUntil >= utteranceEnd - 0.02
                }
                if !coverageComplete {
                    // A single uncovered realtime utterance means a chunk may be missing or
                    // truncated. Keep this speaker's complete realtime transcript as a safe fallback.
                    batchFallbackSpeakers.insert(speaker)
                    finalLines.append(contentsOf: realtimeLines)
                    let lastRangeEnd = ranges.last?.end
                    if let lastRangeEnd {
                        coverageWarnings.append(String(format: "%@: WAV終端 %.1f秒までの範囲に未保存の発話があり、リアルタイム結果を保持", speaker.rawValue, lastRangeEnd))
                    } else {
                        coverageWarnings.append("\(speaker.rawValue): バッチ音声がなく、リアルタイム結果を保持")
                    }
                } else {
                    finalLines.append(contentsOf: segments.map {
                        TranscriptLine(offset: $0.offset, endOffset: nil, speaker: speaker, text: $0.text)
                    })
                }
            }
            lines = finalLines.sorted { $0.offset < $1.offset }
            let failures = outcome.failuresBySpeaker.map {
                "\($0.key.rawValue): \($0.value)（リアルタイム結果を保持）"
            }
            let warnings = coverageWarnings + failures
            if !warnings.isEmpty {
                batchFailureMessage = warnings
                    .joined(separator: " / ")
            }
        } else if !wasSkipped {
            batchFallbackSpeakers = Set(Speaker.allCases)
        }
        isProcessing = false
        isBatchTranscribing = false
        if outcome != nil { batchProgress = 1 }
        estimatedBatchCompletion = nil
        let realtimeFallbackIssues = [
            failedSegments > 0 ? "リアルタイム推論失敗 \(failedSegments) 区間" : nil,
            droppedAudioSeconds > 0 ? String(format: "リアルタイム推論待ちで破棄 %.0f 秒", droppedAudioSeconds) : nil
        ].compactMap { $0 }
        let issues = [captureFailureMessage, wasSkipped ? nil : batchFailureMessage].compactMap { $0 } + realtimeFallbackIssues
        let selfStats = sttMetrics[.selfUser, default: STTMetrics()]
        let othersStats = sttMetrics[.others, default: STTMetrics()]
        audioDiagnosticSummary += "\nSTT 自分: queued \(selfStats.queued), success \(selfStats.succeeded), empty \(selfStats.empty), failed \(selfStats.failed)"
        audioDiagnosticSummary += "\nSTT 相手: queued \(othersStats.queued), success \(othersStats.succeeded), empty \(othersStats.empty), failed \(othersStats.failed)"
        let batchSummary = wasSkipped
            ? "スキップ（リアルタイム結果を使用）"
            : (batchFailureMessage ?? "完了")
        audioDiagnosticSummary += "\nバッチ: \(batchSummary)"
        if let archive = audioArchive {
            Task.detached(priority: .utility) { archive.remove() }
        }
        audioArchive = nil
        if !issues.isEmpty { status = "部分完了: \(issues.joined(separator: "、"))。保存前に欠落を確認してください。" }
        else if wasSkipped {
            status = lines.isEmpty ? "リアルタイム結果はありませんでした。" : "完了: \(lines.count) 件。リアルタイム結果を使用しています。"
        }
        else {
            status = lines.isEmpty ? "音声から発話を検出できませんでした。" : "完了: \(lines.count) 件。バッチ処理済みのテキストを保存できます。"
        }
    }

    func saveText() {
        guard !lines.isEmpty, !isAwaitingBatchChoice else { return }
        let panel = NSSavePanel()
        let filenameFormatter = DateFormatter()
        filenameFormatter.locale = Locale(identifier: "en_US_POSIX")
        filenameFormatter.timeZone = .current
        filenameFormatter.dateFormat = "yyyy-MM-dd_HH-mm-ss"
        panel.nameFieldStringValue = "会議文字起こし-\(filenameFormatter.string(from: sessionStartedAt)).txt"
        panel.allowedContentTypes = [.plainText]
        guard panel.runModal() == .OK, let url = panel.url else { return }
        let realtimeFallbackIssues = [
            failedSegments > 0 ? "リアルタイム推論失敗区間: \(failedSegments)" : nil,
            droppedAudioSeconds > 0 ? String(format: "リアルタイム推論待ち上限で破棄: 約 %.0f 秒", droppedAudioSeconds) : nil
        ].compactMap { $0 }
        let warnings = [captureFailureMessage, batchFailureMessage].compactMap { $0 } + realtimeFallbackIssues
        let warningHeader = warnings.isEmpty ? "" : "# 注意: この文字起こしは一部欠落している可能性があります\n# \(warnings.joined(separator: " / "))\n\n"
        let text = warningHeader + lines.sorted { $0.offset < $1.offset }.map { "[\($0.timeLabel)] \($0.speaker.rawValue): \($0.text)" }.joined(separator: "\n") + "\n"
        do { try text.write(to: url, atomically: true, encoding: .utf8); status = "保存しました: \(url.lastPathComponent)" }
        catch { status = "テキストを保存できません: \(error.localizedDescription)" }
    }

    private func startMeetingMonitor() {
        detectorTask = Task { [weak self] in
            while !Task.isCancelled {
                try? await Task.sleep(for: .seconds(6))
                guard let self, self.autoDetect, !self.isStarting, !self.isStopping, !self.isProcessing, !self.isAwaitingBatchChoice else { continue }
                let visible = await Self.hasMeetingWindow()
                if visible && !self.meetingWasVisible && !self.isRecording && !self.dismissedCandidate {
                    self.meetingPrompt = "会議らしいウィンドウを検出しました。文字起こしを開始しますか？"
                    self.dismissedCandidate = true
                } else if !visible && self.meetingWasVisible && self.isRecording && !self.dismissedEnd {
                    self.meetingPrompt = "会議ウィンドウが閉じたようです。文字起こしを終了しますか？"
                    self.dismissedEnd = true
                }
                if !visible { self.dismissedCandidate = false }
                if visible { self.dismissedEnd = false }
                self.meetingWasVisible = visible
            }
        }
    }

    private static func hasMeetingWindow() async -> Bool {
        guard CGPreflightScreenCaptureAccess() else { return false }
        guard let content = try? await SCShareableContent.excludingDesktopWindows(false, onScreenWindowsOnly: true) else { return false }
        let identifiers = ["us.zoom.xos", "com.microsoft.teams", "com.microsoft.teams2", "com.google.Chrome", "com.microsoft.edgemac", "com.apple.Safari", "org.mozilla.firefox"]
        let meetingWords = ["zoom meeting", "zoom ミーティング", "google meet", "meet -", "meet.google.com", "teams meeting", "teams.microsoft.com"]
        let apps = Set(content.applications.filter { identifiers.contains($0.bundleIdentifier) }.map(\.processID))
        return content.windows.contains { window in
            guard let title = window.title?.lowercased(), !title.isEmpty else { return false }
            return apps.contains(window.owningApplication?.processID ?? -1) && meetingWords.contains { title.contains($0) }
        }
    }

    nonisolated private static func validateBackend(modelPath: URL, whisperPath: URL) throws {
        let base = FileManager.default.temporaryDirectory.appendingPathComponent("MeetingScribe-check-\(UUID().uuidString)")
        let wav = base.appendingPathExtension("wav")
        let output = base.appendingPathExtension("txt")
        let log = base.appendingPathExtension("log")
        defer {
            try? FileManager.default.removeItem(at: wav)
            try? FileManager.default.removeItem(at: output)
            try? FileManager.default.removeItem(at: log)
        }
        try writeWAV(Array(repeating: Float(0), count: 8_000), to: wav)
        FileManager.default.createFile(atPath: log.path, contents: nil)
        let logHandle = try FileHandle(forWritingTo: log)
        let process = Process()
        process.executableURL = whisperPath
        process.arguments = ["-m", modelPath.path, "-l", "ja", "-f", wav.path, "-t", "2", "-nt", "-np", "-otxt", "-of", base.path]
        process.standardOutput = logHandle
        process.standardError = logHandle
        let exited = DispatchSemaphore(value: 0)
        process.terminationHandler = { _ in exited.signal() }
        do { try process.run() }
        catch { logHandle.closeFile(); throw error }
        if exited.wait(timeout: .now() + 180) == .timedOut {
            if process.isRunning { process.terminate() }
            if exited.wait(timeout: .now() + 5) == .timedOut {
                _ = Darwin.kill(process.processIdentifier, SIGKILL)
                _ = exited.wait(timeout: .now() + 5)
            }
            logHandle.closeFile()
            throw CaptureError.inference("モデル検証がタイムアウトしました。")
        }
        process.waitUntilExit()
        logHandle.closeFile()
        guard process.terminationStatus == 0, FileManager.default.fileExists(atPath: output.path) else {
            let detail = (try? String(contentsOf: log, encoding: .utf8)) ?? ""
            throw CaptureError.inference(detail.isEmpty ? "モデルまたは whisper-cli を検証できませんでした。" : detail)
        }
    }

    nonisolated private static func writeWAV(_ samples: [Float], to url: URL) throws {
        let payloadSize = UInt32(samples.count * 2)
        var data = Data()
        data.append(contentsOf: Array("RIFF".utf8)); data.appendLE(payloadSize + 36)
        data.append(contentsOf: Array("WAVEfmt ".utf8)); data.appendLE(UInt32(16)); data.appendLE(UInt16(1))
        data.appendLE(UInt16(1)); data.appendLE(UInt32(16_000)); data.appendLE(UInt32(32_000)); data.appendLE(UInt16(2)); data.appendLE(UInt16(16))
        data.append(contentsOf: Array("data".utf8)); data.appendLE(payloadSize)
        for sample in samples {
            let value = Int16(max(-1, min(1, sample)) * Float(Int16.max))
            data.appendLE(UInt16(bitPattern: value))
        }
        try data.write(to: url, options: .atomic)
    }
}

private enum CaptureError: LocalizedError {
    case noDisplay
    case download
    case microphonePermission
    case screenCapturePermission
    case inference(String)
    var errorDescription: String? {
        switch self {
        case .noDisplay: return "画面を取得できません。画面収録権限を確認してください。"
        case .download: return "モデル取得に失敗しました。ネットワークを確認してください。"
        case .microphonePermission: return "マイクアクセスが許可されていません。システム設定の「プライバシーとセキュリティ」>「マイク」でMeeting Scribeを許可してください。"
        case .screenCapturePermission: return "画面収録とシステムオーディオ録音が許可されていません。システム設定でMeeting Scribeを許可し、このアプリを終了してから再起動してください。"
        case .inference(let detail): return detail.isEmpty ? "whisper-cli がエラーを返しました。" : detail
        }
    }
}

private final class MeetingAudioCapture: NSObject, SCStreamOutput, SCStreamDelegate, @unchecked Sendable {
    var stream: SCStream?
    let callbackQueue = DispatchQueue(label: "MeetingScribe.audio", qos: .userInitiated)
    private let lock = NSLock()
    private let emissionGroup = DispatchGroup()
    private let sessionStart: TimeInterval
    private let emit: @MainActor (Speaker, [Float], TimeInterval) -> Void
    private let onError: @MainActor (String) -> Void
    private let onStreamFailure: @MainActor (String) -> Void
    private let archive: SessionAudioArchive
    private var accumulators: [Speaker: SpeechAccumulator] = [.selfUser: SpeechAccumulator(), .others: SpeechAccumulator()]
    private var converters: [Speaker: AVAudioConverter] = [:]
    private var inputFormats: [Speaker: AVAudioFormat] = [:]
    private var lastEndTime: [Speaker: TimeInterval] = [:]
    private var pathMetrics: [Speaker: AudioPathMetrics] = [.selfUser: AudioPathMetrics(), .others: AudioPathMetrics()]
    private var didReportAudioError = false
    private let outputFormat = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: 16_000, channels: 1, interleaved: false)!

    init(sessionStart: TimeInterval,
         emit: @escaping @MainActor (Speaker, [Float], TimeInterval) -> Void,
         onError: @escaping @MainActor (String) -> Void,
         onStreamFailure: @escaping @MainActor (String) -> Void,
         archive: SessionAudioArchive) {
        self.sessionStart = sessionStart
        self.emit = emit
        self.onError = onError
        self.onStreamFailure = onStreamFailure
        self.archive = archive
    }

    func stream(_ stream: SCStream, didOutputSampleBuffer sampleBuffer: CMSampleBuffer, of outputType: SCStreamOutputType) {
        guard sampleBuffer.isValid, outputType == .audio || outputType == .microphone else { return }
        let speaker: Speaker = outputType == .microphone ? .selfUser : .others
        lock.lock()
        pathMetrics[speaker, default: AudioPathMetrics()].callbacks += 1
        lock.unlock()
        let sampleTimestamp = CMTimeGetSeconds(CMSampleBufferGetPresentationTimeStamp(sampleBuffer))
        guard sampleTimestamp.isFinite else { return }
        guard let asbd = sampleBuffer.formatDescription?.audioStreamBasicDescription else {
            reportAudioError("音声形式を読み取れないため、この音声区間を処理できませんでした。")
            return
        }
        var audioDescription = asbd
        guard let inputFormat = AVAudioFormat(streamDescription: &audioDescription) else {
            reportAudioError("音声形式を読み取れないため、この音声区間を処理できませんでした。")
            return
        }
        lock.lock()
        pathMetrics[speaker, default: AudioPathMetrics()].inputFrames += CMSampleBufferGetNumSamples(sampleBuffer)
        pathMetrics[speaker, default: AudioPathMetrics()].sampleRate = inputFormat.sampleRate
        pathMetrics[speaker, default: AudioPathMetrics()].channels = Int(inputFormat.channelCount)
        lock.unlock()
        var samples: [Float] = []
        var failure: String?
        do {
            try sampleBuffer.withAudioBufferList { list, _ in
                guard let input = AVAudioPCMBuffer(pcmFormat: inputFormat, bufferListNoCopy: list.unsafePointer), input.frameLength > 0 else {
                    failure = "音声バッファを読み取れませんでした。"
                    return
                }
                let converter: AVAudioConverter
                if let current = converters[speaker], inputFormats[speaker]?.isEqual(inputFormat) == true {
                converter = current
                } else if let created = AVAudioConverter(from: inputFormat, to: outputFormat) {
                    converters[speaker] = created
                    inputFormats[speaker] = inputFormat
                    converter = created
                } else {
                    failure = "音声形式を16 kHzへ変換できませんでした。"
                    return
                }
                let capacity = AVAudioFrameCount(ceil(Double(input.frameLength) * outputFormat.sampleRate / inputFormat.sampleRate) + 64)
                guard let output = AVAudioPCMBuffer(pcmFormat: outputFormat, frameCapacity: capacity) else {
                    failure = "変換用音声バッファを確保できませんでした。"
                    return
                }
                var supplied = false
                var conversionError: NSError?
                let status = converter.convert(to: output, error: &conversionError) { _, inputStatus in
                    if supplied { inputStatus.pointee = .noDataNow; return nil }
                    supplied = true
                    inputStatus.pointee = .haveData
                    return input
                }
                guard status == .haveData || status == .inputRanDry || status == .endOfStream,
                      let channelData = output.floatChannelData else {
                    failure = conversionError?.localizedDescription ?? "音声サンプルレート変換に失敗しました。"
                    return
                }
                samples = Array(UnsafeBufferPointer(start: channelData[0], count: Int(output.frameLength)))
            }
        } catch {
            reportAudioError(error.localizedDescription)
            return
        }
        if let failure { reportAudioError(failure); return }
        guard !samples.isEmpty else { return }
        let start = max(0, sampleTimestamp - sessionStart)
        archive.append(samples, speaker: speaker, offset: start)
        let finiteSamples = samples.filter(\.isFinite)
        lock.lock()
        pathMetrics[speaker, default: AudioPathMetrics()].convertedFrames += samples.count
        pathMetrics[speaker, default: AudioPathMetrics()].sampleCount += finiteSamples.count
        pathMetrics[speaker, default: AudioPathMetrics()].sumSquares += finiteSamples.reduce(0.0) { $0 + Double($1) * Double($1) }
        pathMetrics[speaker, default: AudioPathMetrics()].peak = max(pathMetrics[speaker, default: AudioPathMetrics()].peak, finiteSamples.reduce(Float.zero) { max($0, abs($1)) })
        lock.unlock()
        let duration = Double(CMSampleBufferGetNumSamples(sampleBuffer)) / inputFormat.sampleRate
        lock.lock()
        var completed: [(samples: [Float], start: TimeInterval)] = []
        if let previousEnd = lastEndTime[speaker], start - previousEnd > 0.25 {
            completed.append(contentsOf: accumulators[speaker]?.flush() ?? [])
        }
        completed.append(contentsOf: accumulators[speaker]?.append(samples, at: start) ?? [])
        pathMetrics[speaker, default: AudioPathMetrics()].segments += completed.count
        lastEndTime[speaker] = start + duration
        completed.forEach { _ in emissionGroup.enter() }
        lock.unlock()
        completed.forEach { item in
            Task { @MainActor [emit, emissionGroup] in
                emit(speaker, item.samples, item.start)
                emissionGroup.leave()
            }
        }
    }

    func stop() async throws {
        guard let stream else { return }
        try await stream.stopCapture()
    }

    func flush() {
        lock.lock()
        var remaining: [(Speaker, [Float], TimeInterval)] = []
        for (speaker, var accumulator) in accumulators {
            let flushed = accumulator.flush()
            pathMetrics[speaker, default: AudioPathMetrics()].segments += flushed.count
            remaining.append(contentsOf: flushed.map { (speaker, $0.samples, $0.start) })
        }
        for speaker in Speaker.allCases { accumulators[speaker] = SpeechAccumulator() }
        converters.removeAll()
        inputFormats.removeAll()
        remaining.forEach { _ in emissionGroup.enter() }
        lock.unlock()
        remaining.forEach { item in
            Task { @MainActor [emit, emissionGroup] in
                emit(item.0, item.1, item.2)
                emissionGroup.leave()
            }
        }
    }

    func flushAfterPendingCallbacks() async {
        await withCheckedContinuation { continuation in
            callbackQueue.async { [self] in
                flush()
                emissionGroup.notify(queue: .main) { continuation.resume() }
            }
        }
    }

    func finishArchive() async -> String? { await archive.finish() }

    func diagnosticSummary() -> String {
        lock.lock()
        let snapshot = pathMetrics
        lock.unlock()
        return Speaker.allCases.map { speaker in
            let metrics = snapshot[speaker, default: AudioPathMetrics()]
            let rms = metrics.sampleCount == 0 ? 0 : sqrt(metrics.sumSquares / Double(metrics.sampleCount))
            let audioValues = String(format: "%.0fHz RMS=%.4f peak=%.4f",
                                     metrics.sampleRate, rms, Double(metrics.peak))
            return "\(speaker.rawValue) cb=\(metrics.callbacks) 入力=\(metrics.inputFrames) 変換=\(metrics.convertedFrames) \(audioValues)/\(metrics.channels)ch 発話=\(metrics.segments)"
        }.joined(separator: " / ")
    }

    func stream(_ stream: SCStream, didStopWithError error: Error) {
        Task { @MainActor [onStreamFailure] in onStreamFailure("音声キャプチャが停止しました: \(error.localizedDescription)") }
    }

    private func reportAudioError(_ message: String) {
        archive.fail(message)
        lock.lock()
        let shouldReport = !didReportAudioError
        didReportAudioError = true
        lock.unlock()
        guard shouldReport else { return }
        Task { @MainActor [onError] in onError(message) }
    }
}

private struct SpeechAccumulator {
    private var samples: [Float] = []
    private var segmentStart: TimeInterval = 0
    private var silenceSamples = 0
    private var isSpeaking = false
    private let threshold: Float = 0.006
    private let silenceLimit = 6_400 // 0.4 seconds at 16 kHz
    private let maxLength = 128_000 // 8 seconds

    mutating func append(_ chunk: [Float], at timestamp: TimeInterval) -> [(samples: [Float], start: TimeInterval)] {
        var completed: [(samples: [Float], start: TimeInterval)] = []
        let chunkRMS = sqrt(chunk.reduce(Float(0)) { $0 + $1 * $1 } / Float(max(1, chunk.count)))
        if !isSpeaking && chunkRMS >= threshold {
            isSpeaking = true
            segmentStart = timestamp
            samples.removeAll(keepingCapacity: true)
        }
        guard isSpeaking else { return [] }
        samples.append(contentsOf: chunk)
        if chunkRMS < threshold { silenceSamples += chunk.count } else { silenceSamples = 0 }
        if silenceSamples >= silenceLimit || samples.count >= maxLength {
            let endIndex = max(0, samples.count - silenceSamples)
                                    if endIndex >= 1_920 { completed.append((Array(samples[..<endIndex]), segmentStart)) }
            samples.removeAll(keepingCapacity: true)
            silenceSamples = 0
            isSpeaking = false
        }
        return completed
    }

    mutating func flush() -> [(samples: [Float], start: TimeInterval)] {
        defer { samples.removeAll(keepingCapacity: true); silenceSamples = 0; isSpeaking = false }
        let endIndex = max(0, samples.count - silenceSamples)
        guard isSpeaking, endIndex >= 1_920 else { return [] }
        return [(Array(samples[..<endIndex]), segmentStart)]
    }
}

private extension Data {
    mutating func appendLE<T: FixedWidthInteger>(_ value: T) {
        var little = value.littleEndian
        Swift.withUnsafeBytes(of: &little) { append(contentsOf: $0) }
    }
}

@main
struct MeetingScribeApp: App {
    @StateObject private var model = AppModel()
    var body: some Scene {
        WindowGroup {
            ContentView(model: model)
                .frame(minWidth: 620, minHeight: 500)
                .alert("会議の確認", isPresented: Binding(get: { model.meetingPrompt != nil }, set: { if !$0 { model.meetingPrompt = nil } })) {
                    if model.isRecording {
                        Button("終了", role: .destructive) { model.meetingPrompt = nil; model.stop() }
                    } else {
                        Button("開始") { model.meetingPrompt = nil; model.start() }
                    }
                    Button("後で", role: .cancel) { model.meetingPrompt = nil }
                } message: { Text(model.meetingPrompt ?? "") }
        }
        .windowResizability(.contentSize)
    }
}

private struct ContentView: View {
    @ObservedObject var model: AppModel
    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            HStack {
                VStack(alignment: .leading, spacing: 3) {
                    Text("Meeting Scribe").font(.title2.bold())
                    Text("日本語・ローカル文字起こし").foregroundStyle(.secondary)
                }
                Spacer()
                Circle().fill(model.isRecording ? .red : .gray.opacity(0.45)).frame(width: 10, height: 10)
                Text(model.isStarting ? "準備中" : model.isStopping ? "停止中" : model.isRecording ? "録音中" : model.isProcessing ? "処理中" : "停止中").font(.callout)
            }
            HStack(spacing: 10) {
                Button { model.start() } label: { Label("開始", systemImage: "record.circle") }
                    .buttonStyle(.borderedProminent).disabled(!model.ready || model.batchDownloadProgress != nil || model.isRecording || model.isProcessing || model.isAwaitingBatchChoice || model.isStarting || model.isStopping)
                Button { model.stop() } label: { Label("終了", systemImage: "stop.circle") }
                    .buttonStyle(.bordered).disabled(!model.isRecording || model.isStopping)
                Button { model.saveText() } label: { Label("テキストを保存", systemImage: "arrow.down.to.line") }
                    .buttonStyle(.bordered).disabled(model.lines.isEmpty || model.isRecording || model.isProcessing || model.isAwaitingBatchChoice || model.isStarting || model.isStopping)
                Spacer()
            }
            if !model.ready || model.batchModelPath == nil {
                GroupBox("初回セットアップ") {
                    VStack(alignment: .leading, spacing: 10) {
                        HStack {
                            Image(systemName: model.modelPath == nil ? "circle" : "checkmark.circle.fill").foregroundStyle(model.modelPath == nil ? Color.secondary : Color.green)
                            Text("リアルタイム Kotoba-Whisper v2 Q5_0（約538 MB）")
                            Spacer()
                            if model.modelPath == nil {
                                Button(model.downloadProgress == nil ? "モデルを取得" : "取得中") { model.downloadModel() }
                                    .disabled(model.downloadProgress != nil || model.batchDownloadProgress != nil)
                            }
                        }
                        if let progress = model.downloadProgress { ProgressView(value: progress).progressViewStyle(.linear) }
                        HStack {
                            Image(systemName: model.whisperPath == nil ? "circle" : "checkmark.circle.fill").foregroundStyle(model.whisperPath == nil ? Color.secondary : Color.green)
                            Text("whisper.cpp whisper-cli")
                            Spacer()
                            Button("実行ファイルを選択…") { model.chooseWhisperCLI() }
                        }
                        HStack {
                            Image(systemName: model.batchModelPath == nil ? "circle" : "checkmark.circle.fill").foregroundStyle(model.batchModelPath == nil ? Color.secondary : Color.green)
                            Text("録音後 Whisper large-v3（約3.1 GB）")
                            Spacer()
                            if model.batchModelPath == nil {
                                Button(model.batchDownloadProgress == nil ? "高精度モデルを取得" : "取得中") { model.downloadBatchModel() }
                                    .disabled(model.batchDownloadProgress != nil || model.downloadProgress != nil || model.isRecording || model.isProcessing)
                            }
                        }
                        if let progress = model.batchDownloadProgress { ProgressView(value: progress).progressViewStyle(.linear) }
                    }.padding(.vertical, 4)
                }
            }
            HStack {
                Text(model.status).font(.callout).foregroundStyle(.secondary).lineLimit(2)
                Spacer()
                Toggle("会議候補を通知", isOn: $model.autoDetect).toggleStyle(.switch).labelsHidden()
                Text("自動検知").font(.caption).foregroundStyle(.secondary)
            }
            if model.isBatchTranscribing {
                GroupBox("録音音声を再処理中") {
                    VStack(alignment: .leading, spacing: 8) {
                        HStack {
                            Text(model.batchProgressDetail).font(.caption)
                            Spacer()
                            Text("\(Int(model.batchProgress * 100))%").font(.caption.monospacedDigit()).foregroundStyle(.secondary)
                        }
                        ProgressView(value: model.batchProgress).progressViewStyle(.linear)
                        TimelineView(.periodic(from: .now, by: 1)) { context in
                            Text(model.batchRemainingDescription(at: context.date))
                                .font(.caption)
                                .foregroundStyle(.secondary)
                        }
                    }.padding(.vertical, 2)
                }
            }
            if !model.audioDiagnosticSummary.isEmpty {
                DisclosureGroup("音声診断（音声・文字起こし本文は記録しません）") {
                    ScrollView {
                        Text(model.audioDiagnosticSummary)
                            .font(.caption.monospaced())
                            .textSelection(.enabled)
                            .frame(maxWidth: .infinity, alignment: .leading)
                    }.frame(maxHeight: 110)
                }.font(.caption)
            }
            Divider()
            HStack {
                Text("文字起こし").font(.headline)
                Spacer()
                Text("\(model.lines.count) 件").font(.caption).foregroundStyle(.secondary)
            }
            if model.lines.isEmpty {
                ContentUnavailableView("まだ結果がありません", systemImage: "waveform", description: Text("開始すると「自分」と「相手」を分けて記録します。"))
            } else {
                ScrollViewReader { proxy in
                    ScrollView {
                        LazyVStack(alignment: .leading, spacing: 12) {
                            ForEach(model.lines) { line in
                                HStack(alignment: .top, spacing: 10) {
                                    Text(line.timeLabel).font(.caption.monospacedDigit()).foregroundStyle(.secondary).frame(width: 65, alignment: .leading)
                                    Text(line.speaker.rawValue).font(.caption.bold()).foregroundStyle(line.speaker == .selfUser ? .blue : .orange).frame(width: 42, alignment: .leading)
                                    Text(line.text).textSelection(.enabled).frame(maxWidth: .infinity, alignment: .leading)
                                }.id(line.id)
                            }
                        }.padding(.vertical, 6)
                    }
                }
            }
            Text("「自分」=マイク、「相手」=システム音声（他アプリの音声も含む）。ヘッドホン推奨。マイクは会議アプリのミュートと連動しません。")
                .font(.caption2).foregroundStyle(.tertiary)
        }
        .alert("録音後の処理", isPresented: $model.isAwaitingBatchChoice) {
            Button("Whisper large-v3で再処理") { model.chooseBatchTranscription() }
            Button("リアルタイム結果を使う", role: .cancel) { model.chooseRealtimeOnly() }
        } message: {
            Text("高精度モデルで録音音声を再認識します。長い録音では完了まで時間がかかります。")
        }
        .padding(20)
    }
}
