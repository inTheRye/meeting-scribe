import Foundation
import Darwin

/// Temporary, timestamp-aligned PCM archive used by the post-recording pass.
/// Only a small callback queue is retained in memory; audio is written as it arrives.
final class SessionAudioArchive: @unchecked Sendable {
    static let sampleRate = 16_000
    static let chunkDuration: TimeInterval = 300
    private static let maximumQueuedSeconds: TimeInterval = 10
    private static let minimumDiscardedOverlapBeforeFailure: Int64 = Int64(sampleRate * 5)
    private static let maximumDiscardedOverlapRatio = 0.005
    private static let timestampRegressionTolerance: TimeInterval = 0.25
    private static let staleAge: TimeInterval = 7 * 24 * 60 * 60

    let directory: URL
    private let queue = DispatchQueue(label: "MeetingScribe.audio-archive", qos: .utility)
    private let stateLock = NSLock()
    private var queuedFrames = 0
    private var failureMessage: String?
    private var capturedSpeakers = Set<Speaker>()
    private var receivedFrames: [Speaker: Int64] = [:]
    private var writtenFrames: [Speaker: Int64] = [:]
    private var discardedOverlapFrames: [Speaker: Int64] = [:]
    private var firstInputOffset: [Speaker: TimeInterval] = [:]
    private var lastInputEndOffset: [Speaker: TimeInterval] = [:]
    private var tracks: [Speaker: PCMTrackWriter] = [:]
    private var isClosed = false

    init() throws {
        let processID = ProcessInfo.processInfo.processIdentifier
        directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("MeetingScribe-session-\(processID)-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(
            at: directory,
            withIntermediateDirectories: false,
            attributes: [.posixPermissions: 0o700]
        )
    }

    func append(_ samples: [Float], speaker: Speaker, offset: TimeInterval) {
        guard !samples.isEmpty, offset.isFinite, offset >= 0 else { return }
        stateLock.lock()
        guard !isClosed, failureMessage == nil else { stateLock.unlock(); return }
        if let previousEnd = lastInputEndOffset[speaker], offset < previousEnd - Self.timestampRegressionTolerance {
            failureMessage = String(format: "%@: 音声時刻が %.2f 秒戻りました。バッチ音声を破棄してリアルタイム結果を保持します。", speaker.rawValue, previousEnd - offset)
            stateLock.unlock()
            return
        }
        let limit = Int(Double(Self.sampleRate) * Self.maximumQueuedSeconds)
        guard queuedFrames + samples.count <= limit else {
            failureMessage = "一時音声の書き込みが追いつかず、バッチ音声を保存できませんでした。"
            stateLock.unlock()
            return
        }
        queuedFrames += samples.count
        capturedSpeakers.insert(speaker)
        receivedFrames[speaker, default: 0] += Int64(samples.count)
        firstInputOffset[speaker] = min(firstInputOffset[speaker] ?? offset, offset)
        lastInputEndOffset[speaker] = max(lastInputEndOffset[speaker] ?? 0, offset + Double(samples.count) / Double(Self.sampleRate))
        stateLock.unlock()

        queue.async { [weak self] in
            guard let self else { return }
            do {
                var writer = self.tracks[speaker] ?? PCMTrackWriter(directory: self.directory, stem: speaker.archiveStem)
                let overlaps = try writer.append(samples, at: Int64((offset * Double(Self.sampleRate)).rounded()))
                self.tracks[speaker] = writer
                self.stateLock.lock()
                self.discardedOverlapFrames[speaker, default: 0] += overlaps
                self.writtenFrames[speaker, default: 0] += Int64(samples.count) - overlaps
                let discarded = self.discardedOverlapFrames[speaker, default: 0]
                let received = max(1, self.receivedFrames[speaker, default: 0])
                let discardedRatio = Double(discarded) / Double(received)
                if discarded > Self.minimumDiscardedOverlapBeforeFailure && discardedRatio > Self.maximumDiscardedOverlapRatio {
                    let seconds = Double(discarded) / Double(Self.sampleRate)
                    self.failureMessage = String(format: "%@: 重複する時刻の音声を合計%.1f秒破棄しました。バッチ音声を破棄してリアルタイム結果を保持します。", speaker.rawValue, seconds)
                }
                self.stateLock.unlock()
            } catch {
                self.stateLock.lock()
                self.failureMessage = error.localizedDescription
                self.stateLock.unlock()
            }
            self.stateLock.lock()
            self.queuedFrames = max(0, self.queuedFrames - samples.count)
            self.stateLock.unlock()
        }
    }

    func fail(_ message: String) {
        stateLock.lock()
        if failureMessage == nil { failureMessage = message }
        stateLock.unlock()
    }

    /// Drains queued writes, finalizes WAV headers and returns any archival failure.
    func finish() async -> String? {
        markClosed()
        return await withCheckedContinuation { continuation in
            queue.async {
                for speaker in Speaker.allCases {
                    guard var writer = self.tracks[speaker] else { continue }
                    do {
                        try writer.finish()
                        self.tracks[speaker] = writer
                    } catch {
                        self.stateLock.lock()
                        if self.failureMessage == nil { self.failureMessage = error.localizedDescription }
                        self.stateLock.unlock()
                    }
                }
                self.stateLock.lock()
                let result = self.failureMessage
                self.stateLock.unlock()
                continuation.resume(returning: result)
            }
        }
    }

    private func markClosed() {
        stateLock.lock()
        isClosed = true
        stateLock.unlock()
    }

    var speakersWithInput: Set<Speaker> {
        stateLock.lock()
        defer { stateLock.unlock() }
        return capturedSpeakers
    }

    func diagnosticSummary() -> String {
        stateLock.lock()
        let receivedFrames = self.receivedFrames
        let writtenFrames = self.writtenFrames
        let discardedOverlapFrames = self.discardedOverlapFrames
        let firstInputOffset = self.firstInputOffset
        let lastInputEndOffset = self.lastInputEndOffset
        stateLock.unlock()
        return Speaker.allCases.map { speaker in
            let received = Double(receivedFrames[speaker, default: 0]) / Double(Self.sampleRate)
            let written = Double(writtenFrames[speaker, default: 0]) / Double(Self.sampleRate)
            let discarded = Double(discardedOverlapFrames[speaker, default: 0]) / Double(Self.sampleRate)
            let start = firstInputOffset[speaker].map(Self.clockLabel) ?? "なし"
            let end = lastInputEndOffset[speaker].map(Self.clockLabel) ?? "なし"
            let files = wavFiles(for: speaker)
            let fileSummary = files.isEmpty ? "なし" : files.map { index, duration in
                String(format: "%lld:%.1f秒", index, duration)
            }.joined(separator: ",")
            let fileEnd = files.last.map { Double($0.0) * Self.chunkDuration + $0.1 }
            let fileEndLabel = fileEnd.map(Self.clockLabel) ?? "なし"
            return String(format: "%@: input %.1f秒 %@〜%@ / 実WAV %@ (終端%@) / 入力サンプル %.1f秒 / 重複破棄 %.2f秒", speaker.rawValue, received, start, end, fileSummary, fileEndLabel, written, discarded)
        }.joined(separator: " / ")
    }

    private func wavFiles(for speaker: Speaker) -> [(Int64, TimeInterval)] {
        let prefix = speaker.archiveStem + "-"
        let urls = (try? FileManager.default.contentsOfDirectory(at: directory, includingPropertiesForKeys: [.fileSizeKey])) ?? []
        return urls.compactMap { url in
            let size = (try? url.resourceValues(forKeys: [.fileSizeKey]))?.fileSize
            guard url.pathExtension == "wav",
                  url.deletingPathExtension().lastPathComponent.hasPrefix(prefix),
                  let index = Int64(url.deletingPathExtension().lastPathComponent.dropFirst(prefix.count)),
                  let size, size >= 44 else { return nil }
            return (index, Double(size - 44) / Double(Self.sampleRate * 2))
        }.sorted { $0.0 < $1.0 }
    }

    private static func clockLabel(_ offset: TimeInterval) -> String {
        String(format: "%02d:%05.2f", Int(offset) / 60, offset.truncatingRemainder(dividingBy: 60))
    }

    func remove() {
        try? FileManager.default.removeItem(at: directory)
    }

    static func removeAbandonedArchives() {
        let temporaryDirectory = FileManager.default.temporaryDirectory
        guard let entries = try? FileManager.default.contentsOfDirectory(
            at: temporaryDirectory,
            includingPropertiesForKeys: [.isDirectoryKey, .contentModificationDateKey]
        ) else { return }
        let now = Date()
        for entry in entries where entry.lastPathComponent.hasPrefix("MeetingScribe-session-") {
            let components = entry.lastPathComponent.split(separator: "-")
            guard components.count >= 4,
                  let pid = Int32(components[2]),
                  let values = try? entry.resourceValues(forKeys: [.isDirectoryKey, .contentModificationDateKey]),
                  values.isDirectory == true,
                  let modified = values.contentModificationDate,
                  now.timeIntervalSince(modified) > staleAge else { continue }
            if kill(pid, 0) == 0 || errno != ESRCH { continue }
            try? FileManager.default.removeItem(at: entry)
        }
    }
}

private struct PCMTrackWriter {
    private let directory: URL
    private let stem: String
    private let chunkFrames = Int64(SessionAudioArchive.sampleRate * Int(SessionAudioArchive.chunkDuration))
    private var chunkIndex: Int64?
    private var file: FileHandle?
    private var currentURL: URL?
    private var framesWritten: Int64 = 0
    private var dataBytes: UInt32 = 0
    private var hasSignal = false

    init(directory: URL, stem: String) {
        self.directory = directory
        self.stem = stem
    }

    mutating func append(_ samples: [Float], at absoluteFrame: Int64) throws -> Int64 {
        var sourceIndex = 0
        var frame = absoluteFrame
        var discardedOverlap: Int64 = 0
        while sourceIndex < samples.count {
            let wantedChunk = frame / chunkFrames
            let localFrame = frame % chunkFrames
            if chunkIndex != wantedChunk {
                if let current = chunkIndex, wantedChunk < current {
                    throw ArchiveError.outOfOrderAudio
                }
                if chunkIndex != nil { try closeCurrent(padToChunkEnd: true) }
                try open(chunk: wantedChunk)
            }
            if localFrame > framesWritten {
                try writeSilence(Int(localFrame - framesWritten))
            } else if localFrame < framesWritten {
                let overlap = min(Int64(samples.count - sourceIndex), framesWritten - localFrame)
                discardedOverlap += overlap
                sourceIndex += Int(overlap)
                frame += overlap
                continue
            }
            let count = min(samples.count - sourceIndex, Int(chunkFrames - localFrame))
            try write(samples[sourceIndex..<(sourceIndex + count)])
            sourceIndex += count
            frame += Int64(count)
        }
        return discardedOverlap
    }

    mutating func finish() throws {
        if chunkIndex != nil { try closeCurrent(padToChunkEnd: false) }
    }

    private mutating func open(chunk: Int64) throws {
        let url = directory.appendingPathComponent(String(format: "%@-%08lld.wav", stem, chunk))
        guard FileManager.default.createFile(atPath: url.path, contents: nil, attributes: [.posixPermissions: 0o600]) else {
            throw ArchiveError.cannotCreateAudioFile
        }
        let handle = try FileHandle(forWritingTo: url)
        try handle.write(contentsOf: Self.wavHeader(dataBytes: 0))
        file = handle
        currentURL = url
        chunkIndex = chunk
        framesWritten = 0
        dataBytes = 0
        hasSignal = false
    }

    private mutating func closeCurrent(padToChunkEnd: Bool) throws {
        guard let handle = file else { return }
        if padToChunkEnd, framesWritten < chunkFrames {
            try writeSilence(Int(chunkFrames - framesWritten))
        }
        try handle.seek(toOffset: 4)
        try handle.write(contentsOf: Self.littleEndian(dataBytes + 36))
        try handle.seek(toOffset: 40)
        try handle.write(contentsOf: Self.littleEndian(dataBytes))
        try handle.close()
        if !hasSignal, let currentURL { try? FileManager.default.removeItem(at: currentURL) }
        file = nil
        currentURL = nil
        chunkIndex = nil
        framesWritten = 0
        dataBytes = 0
        hasSignal = false
    }

    private mutating func write(_ samples: ArraySlice<Float>) throws {
        guard let file, !samples.isEmpty else { return }
        var bytes = Data(capacity: samples.count * 2)
        for sample in samples {
            let clean = sample.isFinite ? max(-1, min(1, sample)) : 0
            if abs(clean) >= 0.001 { hasSignal = true }
            let pcm = Int16(clean * Float(Int16.max))
            var value = UInt16(bitPattern: pcm).littleEndian
            Swift.withUnsafeBytes(of: &value) { bytes.append(contentsOf: $0) }
        }
        try file.write(contentsOf: bytes)
        framesWritten += Int64(samples.count)
        dataBytes += UInt32(bytes.count)
    }

    private mutating func writeSilence(_ frameCount: Int) throws {
        guard let file, frameCount > 0 else { return }
        var remaining = frameCount
        let zeros = Data(repeating: 0, count: 64 * 1024)
        while remaining > 0 {
            let bytes = min(remaining * 2, zeros.count)
            try file.write(contentsOf: zeros.prefix(bytes))
            let writtenFrames = bytes / 2
            framesWritten += Int64(writtenFrames)
            dataBytes += UInt32(bytes)
            remaining -= writtenFrames
        }
    }

    private static func wavHeader(dataBytes: UInt32) -> Data {
        var data = Data("RIFF".utf8)
        data.append(littleEndian(dataBytes + 36))
        data.append(Data("WAVEfmt ".utf8))
        data.append(littleEndian(UInt32(16)))
        data.append(littleEndian(UInt16(1)))
        data.append(littleEndian(UInt16(1)))
        data.append(littleEndian(UInt32(SessionAudioArchive.sampleRate)))
        data.append(littleEndian(UInt32(SessionAudioArchive.sampleRate * 2)))
        data.append(littleEndian(UInt16(2)))
        data.append(littleEndian(UInt16(16)))
        data.append(Data("data".utf8))
        data.append(littleEndian(dataBytes))
        return data
    }

    private static func littleEndian<T: FixedWidthInteger>(_ value: T) -> Data {
        var little = value.littleEndian
        return Swift.withUnsafeBytes(of: &little) { Data($0) }
    }
}

private enum ArchiveError: LocalizedError {
    case cannotCreateAudioFile
    case outOfOrderAudio

    var errorDescription: String? {
        switch self {
        case .cannotCreateAudioFile: return "一時音声ファイルを作成できませんでした。"
        case .outOfOrderAudio: return "音声の時刻が順序どおりでないため保存できませんでした。"
        }
    }
}

extension Speaker {
    var archiveStem: String { self == .selfUser ? "microphone" : "system" }
}
