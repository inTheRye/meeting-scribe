import Foundation
import Darwin

struct BatchTranscriptSegment {
    let offset: TimeInterval
    let text: String
}

struct BatchAudioRange {
    let start: TimeInterval
    let end: TimeInterval
}

struct BatchProgressUpdate {
    let fractionCompleted: Double
    let speaker: Speaker?
    let chunkNumber: Int
    let totalChunks: Int
    let chunkFraction: Double
}

struct BatchTranscriptionOutcome {
    let segmentsBySpeaker: [Speaker: [BatchTranscriptSegment]]
    let failuresBySpeaker: [Speaker: String]
    let audioRangesBySpeaker: [Speaker: [BatchAudioRange]]
}

enum BatchTranscriber {
    private static let processTimeout: TimeInterval = 30 * 60

    static func transcribe(
        archiveDirectory: URL,
        capturedSpeakers: Set<Speaker>,
        modelURL: URL,
        whisperURL: URL,
        progress: @escaping (BatchProgressUpdate) -> Void
    ) -> BatchTranscriptionOutcome {
        var results: [Speaker: [BatchTranscriptSegment]] = [:]
        var failures: [Speaker: String] = [:]
        var audioRanges: [Speaker: [BatchAudioRange]] = [:]
        let chunks: [URL]
        do {
            chunks = try FileManager.default.contentsOfDirectory(
                at: archiveDirectory,
                includingPropertiesForKeys: [.isRegularFileKey]
            ).filter { $0.pathExtension == "wav" }
        } catch {
            let message = "一時音声ファイルを読み取れませんでした: \(error.localizedDescription)"
            for speaker in Speaker.allCases { failures[speaker] = message }
            progress(BatchProgressUpdate(fractionCompleted: 1, speaker: nil, chunkNumber: 0, totalChunks: 0, chunkFraction: 1))
            return BatchTranscriptionOutcome(segmentsBySpeaker: results, failuresBySpeaker: failures, audioRangesBySpeaker: audioRanges)
        }

        struct ChunkJob {
            let speaker: Speaker
            let url: URL
            let index: Int64
            let duration: TimeInterval
        }

        var jobs: [ChunkJob] = []
        for speaker in Speaker.allCases {
            let speakerChunks = chunks.filter { $0.lastPathComponent.hasPrefix(speaker.archiveStem + "-") }
                .sorted { $0.lastPathComponent < $1.lastPathComponent }
            guard !speakerChunks.isEmpty else {
                if capturedSpeakers.contains(speaker) {
                    results[speaker] = []
                } else {
                    failures[speaker] = "この話者の音声を保存できませんでした。"
                }
                continue
            }
            for url in speakerChunks {
                do {
                    let index = try chunkIndex(from: url)
                    jobs.append(ChunkJob(speaker: speaker, url: url, index: index, duration: try audioDuration(of: url)))
                } catch {
                    failures[speaker] = error.localizedDescription
                }
            }
        }

        let totalDuration = max(0.01, jobs.reduce(0) { $0 + max(0.01, $1.duration) })
        var completedDuration = 0.0
        let totalChunks = jobs.count
        if totalChunks == 0 {
            progress(BatchProgressUpdate(fractionCompleted: 1, speaker: nil, chunkNumber: 0, totalChunks: 0, chunkFraction: 1))
        }
        for (jobIndex, job) in jobs.enumerated() {
            let chunkNumber = jobIndex + 1
            let weight = max(0.01, job.duration)
            let chunkOffset = Double(job.index) * SessionAudioArchive.chunkDuration
            let chunkEnd = chunkOffset + job.duration
            audioRanges[job.speaker, default: []].append(BatchAudioRange(start: chunkOffset, end: chunkEnd))
            let completedDurationBeforeChunk = completedDuration
            let reportChunkProgress: (Double) -> Void = { chunkFraction in
                let fraction = min(1, max(0, (completedDurationBeforeChunk + weight * chunkFraction) / totalDuration))
                progress(BatchProgressUpdate(
                    fractionCompleted: fraction,
                    speaker: job.speaker,
                    chunkNumber: chunkNumber,
                    totalChunks: totalChunks,
                    chunkFraction: min(1, max(0, chunkFraction))
                ))
            }
            reportChunkProgress(0)
            if failures[job.speaker] == nil {
                do {
                    let segments = try transcribeChunk(
                        job.url,
                        offset: chunkOffset,
                        modelURL: modelURL,
                        whisperURL: whisperURL,
                        outputDirectory: archiveDirectory,
                        progress: reportChunkProgress
                    )
                    results[job.speaker, default: []].append(contentsOf: segments)
                } catch {
                    failures[job.speaker] = error.localizedDescription
                    results.removeValue(forKey: job.speaker)
                }
            }
            completedDuration += weight
            reportChunkProgress(1)
        }
        for speaker in Speaker.allCases where capturedSpeakers.contains(speaker) && failures[speaker] == nil && results[speaker] == nil {
            results[speaker] = []
        }
        for speaker in results.keys { results[speaker]?.sort { $0.offset < $1.offset } }
        return BatchTranscriptionOutcome(segmentsBySpeaker: results, failuresBySpeaker: failures, audioRangesBySpeaker: audioRanges)
    }

    private static func audioDuration(of url: URL) throws -> TimeInterval {
        let fileSize = (try url.resourceValues(forKeys: [.fileSizeKey])).fileSize ?? 0
        guard fileSize >= 44, (fileSize - 44).isMultiple(of: 2) else {
            throw BatchTranscriptionError.invalidAudioFile
        }
        return Double(fileSize - 44) / Double(SessionAudioArchive.sampleRate * 2)
    }

    private static func transcribeChunk(
        _ audioURL: URL,
        offset: TimeInterval,
        modelURL: URL,
        whisperURL: URL,
        outputDirectory: URL,
        progress: @escaping (Double) -> Void
    ) throws -> [BatchTranscriptSegment] {
        let fileSize = (try audioURL.resourceValues(forKeys: [.fileSizeKey])).fileSize ?? 0
        guard fileSize >= 44, (fileSize - 44).isMultiple(of: 2) else {
            throw BatchTranscriptionError.invalidAudioFile
        }
        let durationMilliseconds = (fileSize - 44) * 1_000 / (SessionAudioArchive.sampleRate * 2)
        let base = outputDirectory.appendingPathComponent("batch-\(UUID().uuidString)")
        let jsonURL = base.appendingPathExtension("json")
        let logURL = base.appendingPathExtension("log")
        defer {
            try? FileManager.default.removeItem(at: jsonURL)
            try? FileManager.default.removeItem(at: logURL)
        }
        guard FileManager.default.createFile(atPath: logURL.path, contents: nil, attributes: [.posixPermissions: 0o600]) else {
            throw BatchTranscriptionError.cannotCreateLog
        }
        let log = try FileHandle(forWritingTo: logURL)
        let process = Process()
        process.executableURL = whisperURL
        process.arguments = [
            "-m", modelURL.path,
            "-l", "ja",
            "-f", audioURL.path,
            "-t", "4",
            "-np",
            "--print-progress",
            "-oj",
            "-of", base.path
        ]
        process.standardOutput = log
        let progressPipe = Pipe()
        let progressReader = WhisperProgressReader(handle: progressPipe.fileHandleForReading, onProgress: progress)
        let canReadProgress = progressReader.start()
        process.standardError = canReadProgress ? progressPipe : log
        let exited = DispatchSemaphore(value: 0)
        process.terminationHandler = { _ in exited.signal() }
        do {
            try process.run()
            if canReadProgress { try? progressPipe.fileHandleForWriting.close() }
        } catch {
            if canReadProgress {
                try? progressPipe.fileHandleForWriting.close()
                progressReader.waitUntilFinished()
            }
            log.closeFile()
            throw error
        }
        if exited.wait(timeout: .now() + processTimeout) == .timedOut {
            if process.isRunning { process.terminate() }
            if exited.wait(timeout: .now() + 5) == .timedOut {
                _ = Darwin.kill(process.processIdentifier, SIGKILL)
                _ = exited.wait(timeout: .now() + 5)
            }
            if canReadProgress { progressReader.waitUntilFinished() }
            log.closeFile()
            throw BatchTranscriptionError.timedOut
        }
        process.waitUntilExit()
        if canReadProgress { progressReader.waitUntilFinished() }
        log.closeFile()
        guard process.terminationStatus == 0 else {
            let stdout = (try? String(contentsOf: logURL, encoding: .utf8)) ?? ""
            let detail = ([stdout, canReadProgress ? progressReader.output : ""].filter { !$0.isEmpty })
                .joined(separator: "\n").trimmingCharacters(in: .whitespacesAndNewlines)
            throw BatchTranscriptionError.inference(detail)
        }
        let data = try Data(contentsOf: jsonURL)
        let decoded = try JSONDecoder().decode(WhisperJSON.self, from: data)
        var result: [BatchTranscriptSegment] = []
        for segment in decoded.transcription {
            let text = segment.text.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !text.isEmpty else { continue }
            guard let milliseconds = segment.offsets?.from,
                  milliseconds >= 0,
                  milliseconds <= durationMilliseconds else {
                throw BatchTranscriptionError.invalidTimestamp
            }
            result.append(BatchTranscriptSegment(offset: offset + Double(milliseconds) / 1_000, text: text))
        }
        return result
    }

    private static func chunkIndex(from url: URL) throws -> Int64 {
        let basename = url.deletingPathExtension().lastPathComponent
        guard let value = basename.split(separator: "-").last,
              let index = Int64(value) else { throw BatchTranscriptionError.invalidChunkName }
        return index
    }
}

private final class WhisperProgressReader: @unchecked Sendable {
    private let handle: FileHandle
    private let fileDescriptor: Int32
    private let onProgress: (Double) -> Void
    private let finished = DispatchSemaphore(value: 0)
    private let lock = NSLock()
    private var capturedOutput = ""
    private var cancelled = false
    private var started = false

    var output: String {
        lock.lock()
        defer { lock.unlock() }
        return capturedOutput
    }

    init(handle: FileHandle, onProgress: @escaping (Double) -> Void) {
        self.handle = handle
        self.fileDescriptor = handle.fileDescriptor
        self.onProgress = onProgress
    }

    func start() -> Bool {
        let flags = fcntl(fileDescriptor, F_GETFL)
        guard flags >= 0, fcntl(fileDescriptor, F_SETFL, flags | O_NONBLOCK) == 0 else {
            finished.signal()
            return false
        }
        started = true
        DispatchQueue.global(qos: .utility).async { [self] in
            defer { finished.signal() }
            var pending = ""
            var bytes = [UInt8](repeating: 0, count: 8 * 1024)
            while !isCancelled {
                var descriptor = pollfd(fd: fileDescriptor, events: Int16(POLLIN | POLLHUP | POLLERR), revents: 0)
                let ready = Darwin.poll(&descriptor, 1, 100)
                if ready < 0 {
                    if errno == EINTR { continue }
                    break
                }
                if ready == 0 { continue }
                if descriptor.revents & Int16(POLLNVAL) != 0 { break }
                let count = bytes.withUnsafeMutableBytes { buffer in
                    Darwin.read(fileDescriptor, buffer.baseAddress, buffer.count)
                }
                if count == 0 { break }
                if count < 0 {
                    if errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK { continue }
                    break
                }
                let data = Data(bytes.prefix(Int(count)))
                let text = String(decoding: data, as: UTF8.self)
                lock.lock()
                capturedOutput += text
                if capturedOutput.utf8.count > 64 * 1024 {
                    capturedOutput = String(capturedOutput.suffix(64 * 1024))
                }
                lock.unlock()
                pending += text
                while let newline = pending.firstIndex(of: "\n") {
                    let line = String(pending[..<newline])
                    pending.removeSubrange(...newline)
                    guard let marker = line.range(of: "progress =") else { continue }
                    let digits = line[marker.upperBound...]
                        .drop(while: { !$0.isNumber })
                        .prefix(while: { $0.isNumber })
                    if let percentage = Double(digits), (0...100).contains(percentage) {
                        onProgress(percentage / 100)
                    }
                }
                if pending.utf8.count > 4 * 1024 {
                    pending = String(pending.suffix(4 * 1024))
                }
            }
        }
        return true
    }

    func waitUntilFinished(timeout: TimeInterval = 5) {
        guard finished.wait(timeout: .now() + timeout) == .timedOut else { return }
        lock.lock()
        cancelled = true
        lock.unlock()
        guard finished.wait(timeout: .now() + 1) == .timedOut else { return }
        if started { try? handle.close() }
    }

    private var isCancelled: Bool {
        lock.lock()
        defer { lock.unlock() }
        return cancelled
    }
}

private struct WhisperJSON: Decodable {
    let transcription: [Segment]

    struct Segment: Decodable {
        let offsets: Offsets?
        let text: String
    }

    struct Offsets: Decodable {
        let from: Int?
    }
}

private enum BatchTranscriptionError: LocalizedError {
    case cannotCreateLog
    case invalidChunkName
    case invalidAudioFile
    case invalidTimestamp
    case timedOut
    case inference(String)

    var errorDescription: String? {
        switch self {
        case .cannotCreateLog: return "バッチ処理ログを作成できませんでした。"
        case .invalidChunkName: return "一時音声ファイル名を読み取れませんでした。"
        case .invalidAudioFile: return "一時音声ファイルの長さが不正です。"
        case .invalidTimestamp: return "バッチ文字起こしの時刻情報が不正です。"
        case .timedOut: return "バッチ文字起こしが30分でタイムアウトしました。"
        case .inference(let detail): return detail.isEmpty ? "whisper-cli がバッチ処理に失敗しました。" : detail
        }
    }
}
