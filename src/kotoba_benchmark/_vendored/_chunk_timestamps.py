import difflib
import re
import unicodedata
from dataclasses import dataclass

from more_itertools import flatten


@dataclass
class ChunkTimestampsProcessor:
    delimiter: str = " "
    text_key: str = "word"
    ignore_characters: str = ",.!?;:()'-、。"
    punctuation_regex: re.Pattern = re.compile(r"[、。,.]+", re.UNICODE)

    def is_punct_only(self, text: str) -> str:
        return (
            "".join(
                c for c in text if c not in self.ignore_characters + self.delimiter
            ).strip()
            == ""
        )

    def normalize_chunk_text(self, text: str) -> str:
        text = unicodedata.normalize("NFKC", text)
        text = "".join(
            c for c in text if c not in self.ignore_characters + self.delimiter
        )
        return text.casefold()

    def verify_text_match(
        self,
        text_chunks: list[str],
        timestamps: list[dict],
    ) -> list[dict]:
        # Normalize and combine all text chunks
        full_text = self.normalize_chunk_text("".join(text_chunks))

        accumulated_text = ""
        collected_timestamps = []

        for timestamp in timestamps:
            # Append the normalized text from the current timestamp
            accumulated_text += self.normalize_chunk_text(timestamp[self.text_key])
            collected_timestamps.append(timestamp)

        # Check if we've reached the full text
        return accumulated_text == full_text

    # Split timestamps by punctuation, e.g.:
    # {'end': 12.360000000000127, 'start': 11.080000000000382, 'word': 'で、もし'}
    # →[{'start': 11.080000000000382, 'end': 12.360000000000127, 'word': 'で'}, {'start': 12.360000000000127, 'end': 12.360000000000127, 'word': 'もし'}]
    def split_ts(self, ts: dict) -> list[dict]:
        start: float = ts["start"]
        end: float = ts["end"]
        split_words = [
            w for w in self.punctuation_regex.split(ts[self.text_key]) if w.strip()
        ]
        if len(split_words) == 0:
            # Ignore pure punctuation / empty fragments.
            return []
        return [
            {
                "start": start,
                "end": end,
                self.text_key: split_words[0],
            }
        ] + [
            # Create a new timestamp for each subsequent word with zero-duration
            {
                "start": end,
                "end": end,
                self.text_key: word.strip(),
            }
            for word in split_words[1:]
            if word.strip()
        ]

    def aggregate_timestamps(
        self,
        text_chunks: list[str],
        timestamps: list[dict],
    ) -> list[list[float]]:
        """
        `text_chunks` is text in chunks output by ChatGPT
        `timestamps` is the timestamps output by whisper during speech recognition
        Aggregate the timestamps to align with the text chunks.
        """
        timestamps = list(flatten(self.split_ts(ts) for ts in timestamps))

        if len(text_chunks) == 0:
            raise ValueError("No chunks were provided for timestamp aggregation")

        # If timestamps are empty, only allow all-empty chunks; otherwise align is impossible.
        if len(timestamps) == 0:
            if all(not self.normalize_chunk_text(chunk) for chunk in text_chunks):
                return [[0.0, 0.0] for _ in text_chunks]
            raise ValueError("No timestamp tokens available for non-empty chunks")

        chunk_timestamps: list[list[float]] = []  # [[start_time, end_time], ...]

        # Check if text is the same
        full_text_in_chunks = self.normalize_chunk_text("".join(text_chunks))
        full_text_in_timestamps = self.normalize_chunk_text(
            "".join(t[self.text_key] for t in timestamps)
        )

        if full_text_in_chunks != full_text_in_timestamps:
            # Loose validation
            if (
                len(full_text_in_chunks) != len(full_text_in_timestamps)
                or sum(
                    [
                        1
                        for c1, c2 in zip(full_text_in_chunks, full_text_in_timestamps)
                        if c1 != c2
                    ]
                )
                > 5
            ):
                raise ValueError("Mismatched text")

        stack_text_in_chunks = ""
        stack_text_in_timestamps = ""

        start_timestamp_idx, end_timestamp_idx = 0, 0

        for text_chunk in text_chunks:
            # Normalize text_chunk
            normalized_chunk = self.normalize_chunk_text(text_chunk)
            if not normalized_chunk:
                if end_timestamp_idx > 0:
                    boundary = timestamps[end_timestamp_idx - 1]["end"]
                else:
                    boundary = timestamps[0]["start"]
                chunk_timestamps.append([boundary, boundary])
                continue

            stack_text_in_chunks += normalized_chunk

            # Aggregate words in timestamp to match with text_chunk
            while len(stack_text_in_timestamps) < len(stack_text_in_chunks):
                if end_timestamp_idx >= len(timestamps):
                    raise ValueError(
                        "Timestamp alignment exceeded available tokens while processing chunks"
                    )
                stack_text_in_timestamps += self.normalize_chunk_text(
                    timestamps[end_timestamp_idx][self.text_key]
                )
                end_timestamp_idx += 1

            if end_timestamp_idx <= 0:
                raise ValueError(
                    "Timestamp alignment failed: no timestamp consumed for chunk"
                )

            chunk_timestamps.append(
                [
                    timestamps[start_timestamp_idx]["start"],  # Start of chunk
                    timestamps[end_timestamp_idx - 1]["end"],  # End of chunk
                ]
            )

            if len(stack_text_in_timestamps) == len(stack_text_in_chunks):
                start_timestamp_idx: int = end_timestamp_idx
            else:
                # Chunk boundary falls inside a timestamp; overlap the last timestamp.
                start_timestamp_idx = max(end_timestamp_idx - 1, 0)

        if start_timestamp_idx != len(timestamps):
            raise ValueError(f"{start_timestamp_idx=} != {len(timestamps)=}")
        if len(chunk_timestamps) != len(text_chunks):
            last_chunk = chunk_timestamps[-1] if chunk_timestamps else None
            raise ValueError(
                f"{len(chunk_timestamps)=} != {len(text_chunks)=}, last_chunk={last_chunk}"
            )

        return chunk_timestamps

    def ts_with_text(self, timestamps: list, text_chunks: list, with_space: bool):
        assert len(timestamps) == len(text_chunks), (
            f"{len(timestamps)=} != {len(text_chunks)=}"
        )
        assert len(timestamps) > 0, f"{len(timestamps)=}"

        return [
            {"start": ts[0], "end": ts[1], "text": spacing(text, with_space)}
            for ts, text in zip(timestamps, text_chunks)
        ]

    def extract_text_from_timestamps(self, timestamps: list[dict]) -> str:
        """Extract and concatenate all words from timestamps."""
        return "".join(ts["word"] if "word" in ts else ts["text"] for ts in timestamps)

    def fix_chunks_by_edit_distance(
        self,
        src_chunks: list[str],
        timestamps: list[dict],
    ) -> list[str]:
        """
        Use dynamic programming to find optimal alignment between chunks and timestamps.
        This approach minimizes the edit distance while preserving chunk boundaries.
        """
        text_key = "word" if "word" in timestamps[0] else "text"

        # Build a mapping of where each timestamp word starts in the reference text
        timestamp_positions = []
        pos = 0
        for ts in timestamps:
            timestamp_positions.append(pos)
            pos += len(ts[text_key])
        timestamp_positions.append(pos)  # End position

        # For each chunk, find the best matching range in timestamps
        corrected_chunks = []
        current_ts_idx = 0
        best_start_positions = []

        for chunk in src_chunks:
            chunk_clean = chunk.strip()
            best_match_score = -1
            best_start_ts = current_ts_idx
            best_end_ts = current_ts_idx

            # Try different ranges of timestamps
            for start_ts in range(current_ts_idx, len(timestamps)):
                accumulated_text = ""
                for end_ts in range(
                    start_ts, min(start_ts + 50, len(timestamps))
                ):  # Limit search window
                    accumulated_text += timestamps[end_ts][text_key]

                    # Calculate similarity
                    similarity = difflib.SequenceMatcher(
                        None, chunk_clean, accumulated_text
                    ).ratio()

                    # Prefer matches that are close in length
                    length_ratio = min(len(chunk_clean), len(accumulated_text)) / max(
                        len(chunk_clean), len(accumulated_text)
                    )
                    score = similarity * length_ratio

                    if score >= best_match_score:
                        best_match_score = score
                        best_start_ts = start_ts
                        best_end_ts = end_ts + 1

                    # Early termination if accumulated text is much longer than chunk
                    if len(accumulated_text) > len(chunk_clean) * 1.5:
                        break

            # guard against completely unmatched chunks
            if best_match_score < 0:
                raise ValueError(f"No good match found for chunk: '{chunk_clean}'")

            # Extract the matched text from reference
            best_start_positions.append(best_start_ts)
            start_pos = timestamp_positions[best_start_ts]
            end_pos = timestamp_positions[best_end_ts]
            current_ts_idx = best_end_ts

        best_start_positions[0] = 0
        corrected_chunks = []
        for idx in range(len(best_start_positions) - 1):
            chunk = ""
            start_pos = best_start_positions[idx]
            end_pos = best_start_positions[idx + 1]
            for ts in timestamps[start_pos:end_pos]:
                chunk += ts[text_key]
            corrected_chunks.append(chunk.strip())

        last_chunk_texts = [
            ts[text_key] for ts in timestamps[best_start_positions[-1] :]
        ]
        corrected_chunks.append("".join(last_chunk_texts))

        return corrected_chunks


def spacing(text: str, with_space: bool):
    text = text.strip()
    if not text:
        return ""
    if with_space:
        return " " + text
    else:
        return text


def ts_with_text(
    timestamps: list, text_chunks: list, with_space: bool
):  # -> list[dict[str, Any]]:
    assert len(timestamps) == len(text_chunks)
    assert len(timestamps) > 0

    return [
        {"start": ts[0], "end": ts[1], "text": spacing(text, with_space)}
        for ts, text in zip(timestamps, text_chunks)
    ]


def join_ts_text(timestamps: list):
    return "".join([ts["text"] for ts in timestamps])


def ts_to_chunks(english_segments, japanese_segments):
    """
    Converts aligned English and Japanese subtitle segments into a structured format.

    Parameters:
    - english_segments: list of dicts with keys 'start', 'end', and 'text'
    - japanese_segments: same structure as english_segments

    Returns:
    - dict with 'original_translation' and 'original_chunk_pairs'
    """
    min_length = min(len(english_segments), len(japanese_segments))
    english_trimmed = english_segments[:min_length]
    japanese_trimmed = japanese_segments[:min_length]

    # original_translation = "".join([jp["text"].strip() for jp in japanese_trimmed])
    original_chunk_pairs = [
        [en["text"].strip(), jp["text"].strip()]
        for en, jp in zip(english_trimmed, japanese_trimmed)
    ]

    return original_chunk_pairs


if __name__=='__main__':
    src_chunks = ['just to follow me.', 'Are there any or is it a single box?', 'other options you have.', ' For example,', 'Marishi wants with you or uh we acquire the asset by the something far or something like that.', 'If if there are', 'I need to something that', 'but but this this also this is one option.', "I I think it I think it could be, um, and obviously you're already an investor.", 'You are also a strong partner to us.', 'Um, and we do want to support you.', 'Um, we think that the assets and some of the data could be of great value to parties like Marushi,or maybe if the investors were interested in a second Singapore operation.', 'Um, our board has to of course explore all of our alternatives, but including Marushi in that process, I think would be something that we of course would be happy to do.', 'And would want to um rapidly have those conversations as well.', 'I know that you have some board and other considerations, but um we would be happy to look at options with Marushi.', 'Thank you.', 'I think we have nothing more from our side.', 'Good.', 'So thank you for sharing that details data.', 'Um, it is unfortunate, but um, we we would like to continue to have close communications with you.', 'Um, including in the additional data, um after the update us.', 'And also the non-clinical team will continue communicate with IC for any outstanding practices.', 'And of course, uh the BD team will continue to talk with uh closely to Adam and Jane.', 'Okay, so that that sounds good.', "Yeah, we just say I know that you guys are going to be thinking a lot about this data, but we do have our statisticians around for the next month or so, so if you're any you're interested in any statistical analysis like just popping your head that you think might be interesting for us to do.", "Uh, don't hesitate to uh send them bring them to my attention.", "Because there's there's a lot of ways to slice the data.", "Um, I have my theories about the biggest issues that, you know, and ad and I can do a very full analysis of what all the options are, but you know, in terms of re-peting the study and looking at sub populations, you know, I think there's some interesting questions to be asked so if there are questions you have, just feel free to ask and we can look into them.", 'Thank you, David.', 'Mika, one of the group will continue to work with the pharmacovigilance group.', 'Mhm.', 'Um, so that we can wrap up, we sent them a message last Friday on December 19th.', "We did not hear back from them yet, so we will resend a message again this evening and I'll copy you on it as well to be sure that once you all get back from the holidays as well, that we can get going with that DSU.R.", 'Both studies are closed and completed, so we do plan to follow out DSU.R early and not wait to do it in April.', 'Um, but um, I just want to let you know.', 'So we will continue working with pharmacovigilance as well.', 'Yes.', 'Okay, thank you so much.', ' I will keep them I will give them ups as well.', 'Thank you.', 'Wonderful and I think on that last note, um, a big thanks from our team.', "Um, you've been a great partner, we would love to figure out a way to assist you as you think about alternative fund board, um, even through the holidays, not that you need to work, but we will be available.", "James 9 in particular on the corporate side, as you may have questions that come up from your board or other, um, ideas or let's say even diligence questions.", "Just send us us we'll be working through the holidays as well.", "You don't need to know about that.", "But we we're moving quickly with a number of options and uh, like as questions do come up, but just let us know.", 'Thank you so much.', 'Okay, that sounds good.', "Well, thank you everyone, have a good morning, thank you for your time and we'll talk to you soon, have a lovely holiday as well.", 'Thank you.', 'Thank you.', ' Bye.', 'Bye-bye.', '']
    src_ts=[{'end': 2.162, 'start': 0.672, 'text': 'just to follow me.'}, {'end': 6.122, 'start': 2.772, 'text': 'Are there any or is it a single box?'}, {'end': 10.822, 'start': 7.822, 'text': 'other options you have. For example,'}, {'end': 24.472, 'start': 12.722, 'text': 'Marishi wants with you or uh we acquire the asset by the something far or something like that.'}, {'end': 26.542, 'start': 24.632, 'text': 'If if there are'}, {'end': 30.292, 'start': 28.942, 'text': 'I need to something that'}, {'end': 37.402, 'start': 31.282, 'text': 'but but this this also this is one option.'}, {'end': 48.062, 'start': 39.462, 'text': "I I think it I think it could be, um, and obviously you're already an investor."}, {'end': 51.572, 'start': 48.552, 'text': 'You are also a strong partner to us.'}, {'end': 55.102, 'start': 52.422, 'text': 'Um, and we do want to support you.'}, {'end': 65.812, 'start': 55.102, 'text': 'Um, we think that the assets and some of the data could be of great value to parties like Marushi,'}, {'end': 72.162, 'start': 66.402, 'text': 'or maybe if the investors were interested in a second Singapore operation.'}, {'end': 86.392, 'start': 72.982, 'text': 'Um, our board has to of course explore all of our alternatives, but including Marushi in that process, I think would be something that we of course would be happy to do.'}, {'end': 92.962, 'start': 86.69200000000001, 'text': 'And would want to um rapidly have those conversations as well.'}, {'end': 100.912, 'start': 92.962, 'text': 'I know that you have some board and other considerations, but um we would be happy to look at options with Marushi.'}, {'end': 101.864, 'start': 101.444, 'text': 'Thank you.'}, {'end': 109.49199999999999, 'start': 106.402, 'text': 'I think we have nothing more from our side.'}, {'end': 111.27199999999999, 'start': 110.842, 'text': 'Good.'}, {'end': 114.912, 'start': 111.902, 'text': 'So thank you for sharing that details data.'}, {'end': 124.602, 'start': 115.77199999999999, 'text': 'Um, it is unfortunate, but um, we we would like to continue to have close communications with you.'}, {'end': 131.392, 'start': 124.832, 'text': 'Um, including in the additional data, um after the update us.'}, {'end': 141.052, 'start': 131.832, 'text': 'And also the non-clinical team will continue communicate with IC for any outstanding practices.'}, {'end': 148.912, 'start': 141.052, 'text': 'And of course, uh the BD team will continue to talk with uh closely to Adam and Jane.'}, {'end': 152.482, 'start': 150.34199999999998, 'text': 'Okay, so that that sounds good.'}, {'end': 165.982, 'start': 152.952, 'text': "Yeah, we just say I know that you guys are going to be thinking a lot about this data, but we do have our statisticians around for the next month or so, so if you're any you're interested in any statistical analysis like just popping your head that you think might be interesting for us to do."}, {'end': 169.332, 'start': 166.28199999999998, 'text': "Uh, don't hesitate to uh send them bring them to my attention."}, {'end': 172.422, 'start': 169.612, 'text': "Because there's there's a lot of ways to slice the data."}, {'end': 191.662, 'start': 172.882, 'text': "Um, I have my theories about the biggest issues that, you know, and ad and I can do a very full analysis of what all the options are, but you know, in terms of re-peting the study and looking at sub populations, you know, I think there's some interesting questions to be asked so if there are questions you have, just feel free to ask and we can look into them."}, {'end': 192.872, 'start': 192.112, 'text': 'Thank you, David.'}, {'end': 198.796, 'start': 193.916, 'text': 'Mika, one of the group will continue to work with the pharmacovigilance group.'}, {'end': 198.996, 'start': 198.796, 'text': 'Mhm.'}, {'end': 204.716, 'start': 199.086, 'text': 'Um, so that we can wrap up, we sent them a message last Friday on December 19th.'}, {'end': 214.936, 'start': 204.716, 'text': "We did not hear back from them yet, so we will resend a message again this evening and I'll copy you on it as well to be sure that once you all get back from the holidays as well, that we can get going with that DSU.R."}, {'end': 222.846, 'start': 214.936, 'text': 'Both studies are closed and completed, so we do plan to follow out DSU.R early and not wait to do it in April.'}, {'end': 225.89600000000002, 'start': 223.376, 'text': 'Um, but um, I just want to let you know.'}, {'end': 228.516, 'start': 225.89600000000002, 'text': 'So we will continue working with pharmacovigilance as well.'}, {'end': 228.86599999999999, 'start': 228.516, 'text': 'Yes.'}, {'end': 232.836, 'start': 228.86599999999999, 'text': 'Okay, thank you so much. I will keep them I will give them ups as well.'}, {'end': 233.466, 'start': 233.076, 'text': 'Thank you.'}, {'end': 242.501, 'start': 236.541, 'text': 'Wonderful and I think on that last note, um, a big thanks from our team.'}, {'end': 252.781, 'start': 242.501, 'text': "Um, you've been a great partner, we would love to figure out a way to assist you as you think about alternative fund board, um, even through the holidays, not that you need to work, but we will be available."}, {'end': 262.511, 'start': 252.781, 'text': "James 9 in particular on the corporate side, as you may have questions that come up from your board or other, um, ideas or let's say even diligence questions."}, {'end': 266.081, 'start': 262.511, 'text': "Just send us us we'll be working through the holidays as well."}, {'end': 271.431, 'start': 268.411, 'text': "You don't need to know about that."}, {'end': 279.371, 'start': 272.061, 'text': "But we we're moving quickly with a number of options and uh, like as questions do come up, but just let us know."}, {'end': 280.751, 'start': 279.631, 'text': 'Thank you so much.'}, {'end': 282.921, 'start': 281.601, 'text': 'Okay, that sounds good.'}, {'end': 288.661, 'start': 282.921, 'text': "Well, thank you everyone, have a good morning, thank you for your time and we'll talk to you soon, have a lovely holiday as well."}, {'end': 289.521, 'start': 288.661, 'text': 'Thank you.'}, {'end': 291.69100000000003, 'start': 290.411, 'text': 'Thank you. Bye.'}, {'end': 292.821, 'start': 292.091, 'text': 'Bye-bye.'}]
    text_processor = ChunkTimestampsProcessor(text_key="text")
    res = text_processor.aggregate_timestamps(src_chunks, src_ts)
    print(res)
    source_lang = 'en'
    src_aligned = text_processor.ts_with_text(
        text_processor.aggregate_timestamps(src_chunks, src_ts),
        src_chunks,
        with_space=source_lang == "en",
    )
    print(src_aligned)
