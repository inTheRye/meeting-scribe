import Foundation

/// Prevents a technically complete batch pass from replacing a useful live
/// transcript with an empty or obviously collapsed result.
enum BatchTranscriptQualityGuard {
    static func fallbackReason(realtimeTexts: [String], batchSegments: [BatchTranscriptSegment]) -> String? {
        let realtime = normalize(realtimeTexts.joined(separator: " "))
        let batch = normalize(batchSegments.map(\.text).joined(separator: " "))

        guard !realtime.isEmpty else { return nil }
        guard !batch.isEmpty else {
            return "バッチ認識が空だったため、リアルタイム結果を保持"
        }

        if realtime.count >= 40, batch.count * 4 < realtime.count {
            return "バッチ結果がリアルタイム結果より大幅に短いため、リアルタイム結果を保持"
        }

        let batchCharacterCount = batch.count
        let batchRepetitions = repetitionCounts(in: batchSegments.map(\.text))
        let realtimeCharacters = Array(realtime)

        for (phrase, occurrences) in batchRepetitions where occurrences >= 2 {
            let repeatedShare = phrase.count * occurrences
            guard repeatedShare * 2 >= batchCharacterCount else { continue }
            let realtimeCount = countOccurrences(of: Array(phrase), in: realtimeCharacters)
            let minimumBatchCount = realtimeCount == 0 ? 2 : max(3, realtimeCount * 2 + 1)
            guard occurrences >= minimumBatchCount else { continue }
            return "バッチ結果にリアルタイムで確認できない同一文の反復が集中したため、リアルタイム結果を保持"
        }

        return nil
    }

    private static func normalize(_ text: String) -> String {
        String(text.precomposedStringWithCompatibilityMapping.lowercased().unicodeScalars.filter {
            CharacterSet.alphanumerics.contains($0)
        })
    }

    private static func repetitionCounts(in texts: [String]) -> [String: Int] {
        let separators = CharacterSet.whitespacesAndNewlines.union(.punctuationCharacters)
        var counts: [String: Int] = [:]
        for text in texts {
            for rawUnit in text.components(separatedBy: separators) {
                let unit = normalize(rawUnit)
                guard unit.count >= 6 else { continue }
                counts[unit, default: 0] += 1
                for (phrase, occurrences) in repeatedPrefixUnits(in: unit) {
                    counts[phrase, default: 0] += occurrences
                }
            }
        }
        return counts
    }

    /// Finds a repeated phrase that fills most of one punctuation-free unit,
    /// including the common case where Whisper puts many repetitions in one
    /// long segment instead of emitting separate segments. The repeated run
    /// may start anywhere in the segment, after otherwise valid text.
    private static func repeatedPrefixUnits(in text: String) -> [String: Int] {
        let characters = Array(text)
        guard characters.count >= 12 else { return [:] }
        var result: [String: Int] = [:]
        for start in 0...(characters.count - 12) {
            let remaining = characters.count - start
            for phraseLength in 6...min(40, remaining / 2) {
                let phrase = Array(characters[start..<(start + phraseLength)])
                guard !hasShorterPeriod(phrase) else { continue }
                var occurrences = 1
                var index = start + phraseLength
                while index + phraseLength <= characters.count,
                      characters[index..<(index + phraseLength)].elementsEqual(phrase) {
                    occurrences += 1
                    index += phraseLength
                }
                guard occurrences >= 2 else { continue }
                let key = String(phrase)
                result[key] = max(result[key, default: 0], occurrences)
            }
        }
        return result
    }

    private static func hasShorterPeriod(_ phrase: [Character]) -> Bool {
        for period in 1..<6 where phrase.count.isMultiple(of: period) {
            if phrase.indices.allSatisfy({ phrase[$0] == phrase[$0 % period] }) {
                return true
            }
        }
        return false
    }

    private static func countOccurrences(of phrase: [Character], in text: [Character]) -> Int {
        guard !phrase.isEmpty, phrase.count <= text.count else { return 0 }
        var count = 0
        var index = 0
        while index <= text.count - phrase.count {
            if text[index..<(index + phrase.count)].elementsEqual(phrase) {
                count += 1
                index += phrase.count
            } else {
                index += 1
            }
        }
        return count
    }
}
