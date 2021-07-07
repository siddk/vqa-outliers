"""
answers.py

Core script for pre-processing VQA-2 Answer Data --> Truncates Answer Labels based on number of occurrences, and
computes hard-labels.

Reference: https://github.com/hengyuan-hu/bottom-up-attention-vqa/blob/master/tools/compute_softscore.py
"""
import json
import os
import pickle
import re

# CONSTANTS
CONTRACTIONS = {
    "aint": "ain't",
    "arent": "aren't",
    "cant": "can't",
    "couldve": "could've",
    "couldnt": "couldn't",
    "couldn'tve": "couldn't've",
    "couldnt've": "couldn't've",
    "didnt": "didn't",
    "doesnt": "doesn't",
    "dont": "don't",
    "hadnt": "hadn't",
    "hadnt've": "hadn't've",
    "hadn'tve": "hadn't've",
    "hasnt": "hasn't",
    "havent": "haven't",
    "hed": "he'd",
    "hed've": "he'd've",
    "he'dve": "he'd've",
    "hes": "he's",
    "howd": "how'd",
    "howll": "how'll",
    "hows": "how's",
    "Id've": "I'd've",
    "I'dve": "I'd've",
    "Im": "I'm",
    "Ive": "I've",
    "isnt": "isn't",
    "itd": "it'd",
    "itd've": "it'd've",
    "it'dve": "it'd've",
    "itll": "it'll",
    "let's": "let's",
    "maam": "ma'am",
    "mightnt": "mightn't",
    "mightnt've": "mightn't've",
    "mightn'tve": "mightn't've",
    "mightve": "might've",
    "mustnt": "mustn't",
    "mustve": "must've",
    "neednt": "needn't",
    "notve": "not've",
    "oclock": "o'clock",
    "oughtnt": "oughtn't",
    "ow's'at": "'ow's'at",
    "'ows'at": "'ow's'at",
    "'ow'sat": "'ow's'at",
    "shant": "shan't",
    "shed've": "she'd've",
    "she'dve": "she'd've",
    "she's": "she's",
    "shouldve": "should've",
    "shouldnt": "shouldn't",
    "shouldnt've": "shouldn't've",
    "shouldn'tve": "shouldn't've",
    "somebody'd": "somebodyd",
    "somebodyd've": "somebody'd've",
    "somebody'dve": "somebody'd've",
    "somebodyll": "somebody'll",
    "somebodys": "somebody's",
    "someoned": "someone'd",
    "someoned've": "someone'd've",
    "someone'dve": "someone'd've",
    "someonell": "someone'll",
    "someones": "someone's",
    "somethingd": "something'd",
    "somethingd've": "something'd've",
    "something'dve": "something'd've",
    "somethingll": "something'll",
    "thats": "that's",
    "thered": "there'd",
    "thered've": "there'd've",
    "there'dve": "there'd've",
    "therere": "there're",
    "theres": "there's",
    "theyd": "they'd",
    "theyd've": "they'd've",
    "they'dve": "they'd've",
    "theyll": "they'll",
    "theyre": "they're",
    "theyve": "they've",
    "twas": "'twas",
    "wasnt": "wasn't",
    "wed've": "we'd've",
    "we'dve": "we'd've",
    "weve": "we've",
    "werent": "weren't",
    "whatll": "what'll",
    "whatre": "what're",
    "whats": "what's",
    "whatve": "what've",
    "whens": "when's",
    "whered": "where'd",
    "wheres": "where's",
    "whereve": "where've",
    "whod": "who'd",
    "whod've": "who'd've",
    "who'dve": "who'd've",
    "wholl": "who'll",
    "whos": "who's",
    "whove": "who've",
    "whyll": "why'll",
    "whyre": "why're",
    "whys": "why's",
    "wont": "won't",
    "wouldve": "would've",
    "wouldnt": "wouldn't",
    "wouldnt've": "wouldn't've",
    "wouldn'tve": "wouldn't've",
    "yall": "y'all",
    "yall'll": "y'all'll",
    "y'allll": "y'all'll",
    "yall'd've": "y'all'd've",
    "y'alld've": "y'all'd've",
    "y'all'dve": "y'all'd've",
    "youd": "you'd",
    "youd've": "you'd've",
    "you'dve": "you'd've",
    "youll": "you'll",
    "youre": "you're",
    "youve": "you've",
}

MANUAL_MAP = {
    "none": "0",
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
}
ARTICLES = ["a", "an", "the"]
PUNCT = [";", r"/", "[", "]", '"', "{", "}", "(", ")", "=", "+", "\\", "_", "-", ">", "<", "@", "`", ",", "?", "!"]

PERIOD_STRIP = re.compile("(?!<=\d)(\.)(?!\d)")
COMMA_STRIP = re.compile("(\d)(\,)(\d)")


def process_punctuation(inText):
    outText = inText
    for p in PUNCT:
        if (p + " " in inText or " " + p in inText) or (re.search(COMMA_STRIP, inText) != None):
            outText = outText.replace(p, "")
        else:
            outText = outText.replace(p, " ")
    outText = PERIOD_STRIP.sub("", outText, re.UNICODE)
    return outText


def process_digit_article(inText):
    outText = []
    tempText = inText.lower().split()
    for word in tempText:
        word = MANUAL_MAP.setdefault(word, word)
        if word not in ARTICLES:
            outText.append(word)
        else:
            pass
    for wordId, word in enumerate(outText):
        if word in CONTRACTIONS:
            outText[wordId] = CONTRACTIONS[word]
    outText = " ".join(outText)
    return outText


def preprocess_answer(answer):
    answer = process_digit_article(process_punctuation(answer))
    answer = answer.replace(",", "")
    return answer


def get_score(occurrences):
    if occurrences == 0:
        return 0
    elif occurrences == 1:
        return 0.3
    elif occurrences == 2:
        return 0.6
    elif occurrences == 3:
        return 0.9
    else:
        return 1


def filter_answers(answers, min_occurrences=9):
    """Count the number of occurrences of each answer in the train/validation set"""
    occurrence = {}
    for entry in answers:
        ground_truth = preprocess_answer(entry["multiple_choice_answer"])

        # Add Answer, Question ID to Occurrences
        if ground_truth not in occurrence:
            occurrence[ground_truth] = set()
        occurrence[ground_truth].add(entry["question_id"])

    # Iterate through Occurrences and Filter any Answers w/ Fewer than `min_occurrences` occurrences
    for ans in list(occurrence.keys()):
        if len(occurrence[ans]) < min_occurrences:
            occurrence.pop(ans)

    print("\t[*] Filtered Rare Answers --> %d Answers appear >= %d Times..." % (len(occurrence), min_occurrences))
    return occurrence


def create_labels(occurrence):
    """Create Mappings between Answers and Label ID"""
    ans2label, label2ans = {}, []
    for answer in occurrence:
        ans2label[answer] = len(ans2label)
        label2ans.append(answer)
    return ans2label, label2ans


def compute_target(answers, ans2label, label2ans):
    """Associate each example in the dataset with a soft distribution over answers"""
    target = []

    # For simple splits --> treat as multi-class classification (Softmax)
    unlabeled_count = 0
    for entry in answers:
        if entry["multiple_choice_answer"] in ans2label:
            target.append(
                {
                    "question_id": entry["question_id"],
                    "image_id": entry["image_id"],
                    "label": ans2label[entry["multiple_choice_answer"]],
                }
            )
        else:
            # <UNK> Handling
            if "<UNK>" not in ans2label:
                ans2label["<UNK>"] = len(ans2label)
                label2ans.append("<UNK>")

            target.append(
                {"question_id": entry["question_id"], "image_id": entry["image_id"], "label": ans2label["<UNK>"]}
            )
            unlabeled_count += 1
    print("\t[*] # Unanswerable Questions: %d / %d" % (unlabeled_count, len(answers)))

    return target, ans2label, label2ans


def vqa2_create_answers(split="all", vqa2_q="data/VQA2-Questions", cache="data/VQA2-Cache", min_occurrences=9):
    """Create set of possible answers for VQA2 based on Occurrences & Compute Soft-Labels"""
    train_ans = os.path.join(vqa2_q, "v2_mscoco_train2014_annotations.json")
    val_ans = os.path.join(vqa2_q, "v2_mscoco_val2014_annotations.json")

    # Create File Paths and Load from Disk (if exists)
    a2lfile = os.path.join(cache, "%s-ans2label.pkl" % split)
    if os.path.exists(a2lfile):
        with open(a2lfile, "rb") as f:
            ans2label, label2ans = pickle.load(f)

        return ans2label, label2ans

    # Load from JSON
    print("\t[*] Reading Training and Validation Answers for Pre-processing...")
    with open(train_ans, "r") as f:
        train_answers = json.load(f)["annotations"]

    with open(val_ans, "r") as f:
        val_answers = json.load(f)["annotations"]

    # Aggregate and Filter Answers based on Number of Occurrences
    answers = train_answers + val_answers
    occurrence = filter_answers(answers, min_occurrences=min_occurrences)

    # Handle Different Splits
    if split in ["all", "f50", "f60", "f75", "f90"]:
        # Create Answer Labels
        ans2label, label2ans = create_labels(occurrence)

        # Compute Per-Example Target Distribution over Answers (Soft Answers)
        train_target, ans2label, label2ans = compute_target(train_answers, ans2label, label2ans)
        val_target, ans2label, label2ans = compute_target(val_answers, ans2label, label2ans)

        print(
            "\t\t[*] Full Dataset -- %d Train | %d Validation across %d answers..."
            % (len(train_target), len(val_target), len(ans2label))
        )

    elif split == "sports":
        # Manually Define Split based on "most occurring" Sports-related words...
        filtered = [
            "football",
            "soccer",
            "volleyball",
            "basketball",
            "tennis",
            "badminton",
            "baseball",
            "softball",
            "hockey",
            "golf",
            "racing",
            "rugby",
            "boxing",
            "horse racing",
            "swimming",
            "skiing",
            "snowboarding",
            "water skiing",
            "bowling",
            "biking",
        ]

        # Occurrence Filtering
        occurrence = {ans: occurrence[ans] for ans in filtered}

        # Create Answer Labels
        ans2label, label2ans = create_labels(occurrence)

        # Compute Per-Example Target Distribution over Answers (Soft Answers)
        _, ans2label, label2ans = compute_target(train_answers, ans2label, label2ans)
        _, ans2label, label2ans = compute_target(val_answers, ans2label, label2ans)

    elif split == "food":
        # Manually Define Split based on "most occurring" Food-related words...
        filtered = [
            "pizza",
            "sandwich",
            "hot dog",
            "cheese",
            "coffee",
            "fruit",
            "chicken",
            "vegetables",
            "fish",
            "salad",
            "bread",
            "milk",
            "soup",
            "beef",
            "rice",
            "pasta",
            "pork",
            "french fries",
            "cereal",
            "bagel",
        ]

        # Occurrence Filtering
        occurrence = {ans: occurrence[ans] for ans in filtered}

        # Create Answer Labels
        ans2label, label2ans = create_labels(occurrence)

        # Compute Per-Example Target Distribution over Answers (Soft Answers)
        _, ans2label, label2ans = compute_target(train_answers, ans2label, label2ans)
        _, ans2label, label2ans = compute_target(val_answers, ans2label, label2ans)

    else:
        raise NotImplementedError("Split %s not supported!" % split)

    # Dump Ans2Label and Targets to File
    with open(a2lfile, "wb") as f:
        pickle.dump((ans2label, label2ans), f)

    # Return Mapping and Targets
    return ans2label, label2ans
