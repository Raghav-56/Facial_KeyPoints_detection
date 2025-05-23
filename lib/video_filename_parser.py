import os
import argparse
from config.logger_config import logger

EMOTION_DICT = {
    "A": "Anger",
    "D": "Disgust",
    "F": "Fear",
    "H": "Happy",
    "Ha": "Happy",
    "N": "Neutral",
    "S": "Sad",
}

LANGUAGE_DICT = {
    "EN": "English",
    "HI": "Hindi",
}

SPEAKER_NAME_DICT = {
    "A1": "Speaker One",
    "A2": "Speaker Two",
    "A3": "Speaker Three",
    "A4": "Speaker Four",
    "A5": "Speaker Five",
    "A6": "Speaker Six",
    "A7": "Speaker Seven",
    "A8": "Speaker Eight",
    "A9": "Speaker Nine",
    "A10": "Speaker Ten",
    "A11": "Speaker Eleven",
    "A12": "Speaker Twelve",
    "A13": "Speaker Thirteen",
    "A14": "Speaker Fourteen",
    "A15": "Speaker Fifteen",
    "A16": "Speaker Sixteen",
    "A17": "Speaker Seventeen",
    "A18": "Speaker Eighteen",
    "A19": "Speaker Nineteen",
    "A20": "Speaker Twenty",
    "A21": "Speaker Twenty One",
    "A22": "Speaker Twenty Two",
    "A23": "Speaker Twenty Three",
    "A24": "Speaker Twenty Four",
    "A25": "Speaker Twenty Five",
}

SENTENCE_DICT = {
    "S1": "Can't you hear my voice?",
    "S2": "I tried to resolve this issue from my end.",
    "S3": "My electricity bill is not yet updated.",
    "S4": "How much time is needed to update my account details?",
    "S5": "I no longer want to use your services.",
    "S6": "No, I haven't received any updates.",
    "S7": "No, that's fine.",
    "S8": "Okay, but make it quick.",
    "S9": "I am busy. You can call me later.",
    "S10": "Yes, who is calling?",
    "S11": "I hope it will work fine now",
    "S12": "Okay, what do I have to do?",
    "S13": "Fine, send your executive at 10:00 a.m. tomorrow.",
    "S14": "I got the wrong electricity bill.",
    "S15": "Okay, I have all these things ready.",
    "S16": "I have been waiting long to connect.",
    "S17": "Can you fix it fast?",
    "S18": "Well, can you help me?",
}

Gender = {
    "M": [
        "A1",
        "A2",
        "A5",
        "A8",
        "A9",
        "A11",
        "A12",
        "A14",
        "A15",
        "A16",
        "A20",
        "A22",
        "A23",
    ],
    "F": [
        "A3",
        "A4",
        "A6",
        "A7",
        "A10",
        "A13",
        "A17",
        "A18",
        "A19",
        "A21",
        "A24",
        "A25",
    ],
}


def parse_video_filename(filename):
    """
    Parse a structured video filename to extract metadata about speaker, language, emotion, etc.
    """
    base = filename.rsplit(".", 1)[0] if "." in filename else filename

    parts = base.split("_")
    if len(parts) != 4:
        logger.warning(
            "Filename structure unexpected: %s, expected 4 parts separated by underscores",
            filename,
        )
        return {
            "speaker": "unknown",
            "speaker_name": "Unknown Speaker",
            "language": "unknown",
            "language_full": "Unknown Language",
            "emotion": "unknown",
            "emotion_full": "Unknown Emotion",
            "detail": "unknown",
            "sentence": "Unknown sentence",
        }

    speaker, language, emotion_code, detail = [p.strip() for p in parts]

    speaker_name = SPEAKER_NAME_DICT.get(speaker, speaker)
    language_full = LANGUAGE_DICT.get(language, language)
    emotion_full = EMOTION_DICT.get(emotion_code, emotion_code)
    sentence = SENTENCE_DICT.get(detail, detail)

    return {
        "speaker": speaker,
        "speaker_name": speaker_name,
        "language": language,
        "language_full": language_full,
        "emotion": emotion_code,
        "emotion_full": emotion_full,
        "detail": detail,
        "sentence": sentence,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Convert video files to a structured directory hierarchy "
        "based on filename metadata"
    )
    parser.add_argument(
        "--src", required=True, help="Source directory with video files"
    )
    parser.add_argument(
        "--dst", required=True, help="Destination directory for final structure"
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    log_level = 10 if args.verbose else 20
    import logging

    logging.basicConfig(level=log_level, format="%(levelname)s: %(message)s")

    # No convert_structure function is defined, so just print a message
    print("No structure conversion implemented.")


if __name__ == "__main__":
    main()
