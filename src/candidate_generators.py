# -*- coding: utf-8 -*-
from utils import to_lower, asciify
import re
import math
import logging.config

logging.config.fileConfig('resources/logging.ini')
logger = logging.getLogger(__file__)


class TurkishStemSuffixCandidateGenerator(object):

    ROOT_TRANSFORMATION_MAP = {"tıp": "tıb", "prof.": "profesör", "dr.": "doktor",
                               "yi": "ye", "ed": "et", "di": "de"}
    TAG_FLAG_MAP = {0: "Adj", 1: "Adverb", 2: "Conj", 3: "Det", 4: "Dup", 5: "Interj", 6: "Noun", 7: "Postp",
                    8: "Pron", 9: "Ques", 10: "Verb", 11: "Num", 12: "Noun+Prop"}

    SUFFIX_DICT_FILE_PATH = "resources/Suffixes&Tags.txt"
    STEM_LIST_FILE_PATH = "resources/StemListWithFlags.txt"

    CONSONANT_STR = "[bcdfgğhjklmnprsştvyzxwqBCDFGĞHJKLMNPRSŞTVYZXWQ]"
    VOWEL_STR = "[aeıioöuüAEIİOÖUÜ]"
    WIDE_VOWELS_STR = "[aeoöAEOÖ]"
    NARROW_VOWELS_STR = "[uüıiUÜIİ]"

    ENDS_WITH_SOFT_CONSONANTS_REGEX = re.compile(r"^.*[bcdğBCDĞgG]$")
    SUFFIX_TRANSFORMATION_REGEX1 = re.compile(r"[ae]")
    SUFFIX_TRANSFORMATION_REGEX2 = re.compile(r"[ıiuü]")
    ENDS_TWO_CONSONANT_REG = re.compile(r"^.*{}{}$".format(CONSONANT_STR, CONSONANT_STR))
    STARTS_VOWEL_REGEX = re.compile(r"^{}.*$".format(VOWEL_STR))
    ENDS_NARROW_REGEX = re.compile(r"^.*{}$".format(NARROW_VOWELS_STR))
    TAG_SEPARATOR_REGEX = re.compile(r"[\+\^]")
    NON_WORD_REGEX = re.compile(r"^[^A-Za-zışğüçöÜĞİŞÇÖ]+$")

    def __init__(self, case_sensitive=True, asciification=False, suffix_normalization=False):
        self.case_sensitive = case_sensitive
        self.asciification = asciification
        self.suffix_normalization = suffix_normalization
        self.suffix_dic = {}
        logger.info("Reading suffix-tag map...")
        self.read_suffix_dic()
        logger.info("Done.")
        self.stem_dic = {}
        logger.info("Reading stem list")
        self.read_stem_list()
        logger.info("Done.")

    def read_suffix_dic(self):
        with open(TurkishStemSuffixCandidateGenerator.SUFFIX_DICT_FILE_PATH, "r", encoding="UTF-8") as f:
            for line in f:
                splits = line.strip().split("\t")
                suffix = splits[0]
                if suffix not in self.suffix_dic:
                    self.suffix_dic[suffix] = []
                self.suffix_dic[suffix].append(splits[1])
                self.stem_dic = []

    def read_stem_list(self):
        with open(TurkishStemSuffixCandidateGenerator.STEM_LIST_FILE_PATH, "r", encoding="UTF-8") as f:
            for line in f:
                splits = line.strip().split("\t")
                stem = splits[0]
                if not self.case_sensitive:
                    stem = to_lower(stem)
                flag = int(splits[1].strip())
                postags = TurkishStemSuffixCandidateGenerator._parse_flag(flag)
                if stem in self.stem_dic:
                    self.stem_dic[stem] = list(set(list(postags) + self.stem_dic[stem]))
                else:
                    self.stem_dic[stem] = postags

    @staticmethod
    def _parse_flag(flag):
        res = []
        for i in range(12, -1, -1):
            if flag >= math.pow(2, i):
                res.append(TurkishStemSuffixCandidateGenerator.TAG_FLAG_MAP[i])
                flag = flag - math.pow(2, i)
        if flag != 0:
            raise IOError("Error: problems in stem flags!")
        return res

    @staticmethod
    def _transform_soft_consonants(text):
        text = re.sub(r"^(.*)b$", r"\1p", text)
        text = re.sub(r"^(.*)B$", r"\1P", text)
        text = re.sub(r"^(.*)c$", r"\1ç", text)
        text = re.sub(r"^(.*)C$", r"\1Ç", text)
        text = re.sub(r"^(.*)d$", r"\1t", text)
        text = re.sub(r"^(.*)D$", r"\1T", text)
        text = re.sub(r"^(.*)ğ$", r"\1k", text)
        text = re.sub(r"^(.*)Ğ$", r"\1K", text)
        text = re.sub(r"^(.*)g$", r"\1k", text)
        text = re.sub(r"^(.*)G$", r"\1K", text)
        return text

    @staticmethod
    def _root_transform(candidate_roots):
        for i in range(len(candidate_roots)):
            if candidate_roots[i] in TurkishStemSuffixCandidateGenerator.ROOT_TRANSFORMATION_MAP:
                candidate_roots[i] = TurkishStemSuffixCandidateGenerator.ROOT_TRANSFORMATION_MAP[candidate_roots[i]]

    @classmethod
    def suffix_transform(cls, candidate_suffixes):
        for i in range(len(candidate_suffixes)):
            candidate_suffixes[i] = cls.suffix_transform_single(candidate_suffixes[i])

    @classmethod
    def suffix_transform_single(cls, candidate_suffix):
        candidate_suffix = to_lower(candidate_suffix)
        candidate_suffix = cls.SUFFIX_TRANSFORMATION_REGEX1.sub("A", candidate_suffix)
        candidate_suffix = cls.SUFFIX_TRANSFORMATION_REGEX2.sub("H", candidate_suffix)
        return candidate_suffix

    @staticmethod
    def _add_candidate_stem_suffix(stem_candidate, suffix_candidate, candidate_roots, candidate_suffixes):
        # Bana, Sana -> ben, sen
        if stem_candidate == "ban" and suffix_candidate == "a":
            candidate_roots.append("ben")
            candidate_suffixes.append("a")
        elif stem_candidate == "Ban" and suffix_candidate == "a":
            candidate_roots.append("Ben")
            candidate_suffixes.append("a")
        elif stem_candidate == "san" and suffix_candidate == "a":
            candidate_roots.append("sen")
            candidate_suffixes.append("a")
        elif stem_candidate == "San" and suffix_candidate == "a":
            candidate_roots.append("Sen")
            candidate_suffixes.append("a")
        else:
            candidate_roots.append(stem_candidate)
            candidate_suffixes.append(suffix_candidate)
            if len(stem_candidate) > 2 and len(suffix_candidate) > 0 \
                    and stem_candidate[-1] == suffix_candidate[0] \
                    and stem_candidate[-1] in TurkishStemSuffixCandidateGenerator.CONSONANT_STR:
                # CONSONANT DERIVATION
                # his -i > hissi, hak -ı > hakkı, red -i > reddi
                candidate_roots.append(stem_candidate)
                candidate_suffixes.append(suffix_candidate[1:])
            elif len(stem_candidate) > 1 and \
                    TurkishStemSuffixCandidateGenerator.ENDS_NARROW_REGEX.match(stem_candidate) and \
                    "yor" in suffix_candidate:
                # bekle -yor > bekliyor, atla -yor > atliyor
                if stem_candidate.endswith("i") or stem_candidate.endswith("ü"):
                    candidate_roots.append(stem_candidate[:-1] + "e")
                    candidate_suffixes.append(suffix_candidate)
                elif stem_candidate.endswith("ı") or stem_candidate.endswith("u"):
                    candidate_roots.append(stem_candidate[:-1] + "a")
                    candidate_suffixes.append(suffix_candidate)
            if len(stem_candidate) > 2 and \
                    TurkishStemSuffixCandidateGenerator.ENDS_TWO_CONSONANT_REG.match(stem_candidate) and \
                    TurkishStemSuffixCandidateGenerator.STARTS_VOWEL_REGEX.match(suffix_candidate):
                # VOWEL DROP
                # ağız – ım > ağzım, alın –ın – a > alnına
                # burun –u > burnu, bağır –ım > bağrım, beyin –i > beyni
                suffix_start_letter = to_lower(suffix_candidate[0])
                if suffix_start_letter in ["u", "ü", "ı", "i"]:
                    candidate_roots.append(stem_candidate[:-1] + suffix_start_letter + stem_candidate[-1])
                    candidate_suffixes.append(suffix_candidate)
                elif suffix_start_letter == "e":
                    candidate_roots.append(stem_candidate[:-1] + "i" + stem_candidate[-1])
                    candidate_suffixes.append(suffix_candidate)
                    candidate_roots.append(stem_candidate[:-1] + "ü" + stem_candidate[-1])
                    candidate_suffixes.append(suffix_candidate)
                elif suffix_start_letter == "a":
                    candidate_roots.append(stem_candidate[:-1] + "ı" + stem_candidate[-1])
                    candidate_suffixes.append(suffix_candidate)
                    candidate_roots.append(stem_candidate[:-1] + "u" + stem_candidate[-1])
                    candidate_suffixes.append(suffix_candidate)
            if len(stem_candidate) > 2 and TurkishStemSuffixCandidateGenerator.ENDS_WITH_SOFT_CONSONANTS_REGEX.match(stem_candidate):
                # Softening of consonants
                candidate_roots.append(TurkishStemSuffixCandidateGenerator._transform_soft_consonants(stem_candidate))
                candidate_suffixes.append(suffix_candidate)

    def get_stem_suffix_candidates(self, surface_word):
        candidate_roots = []
        candidate_suffixes = []
        for i in range(1, len(surface_word)):
            candidate_root = surface_word[:i]
            candidate_suffix = surface_word[i:]
            if not self.case_sensitive:
                candidate_root = to_lower(candidate_root)
                candidate_suffix = to_lower(candidate_suffix)
            self._add_candidate_stem_suffix(candidate_root, candidate_suffix, candidate_roots, candidate_suffixes)
        if not self.case_sensitive:
            candidate_roots.append(to_lower(surface_word))
        else:
            candidate_roots.append(to_lower(surface_word))
        candidate_suffixes.append("")
        assert len(candidate_roots) == len(candidate_suffixes)
        TurkishStemSuffixCandidateGenerator._root_transform(candidate_roots)
        if self.asciification:
            candidate_roots = [asciify(candidate_root) for candidate_root in candidate_roots]
            candidate_suffixes = [asciify(candidate_suffix) for candidate_suffix in candidate_suffixes]
        if self.suffix_normalization:
            TurkishStemSuffixCandidateGenerator.suffix_transform(candidate_suffixes)
        return candidate_roots, candidate_suffixes

    def _get_tags(self, suffix, stem_tags=None):
        if suffix and len(suffix) > 0:
            if suffix in self.suffix_dic:
                tags = self.suffix_dic[suffix]
                if suffix.startswith("'"):
                    tags += self.suffix_dic[suffix[1:]]
            elif suffix.startswith("'") and suffix[1:] in self.suffix_dic:
                tags = self.suffix_dic[suffix[1:]]
            else:
                return []
        else:
            tags = self.suffix_dic["null"]
        res = []
        tags = list(set(tags))
        for tag in tags:
            tag_sequences = TurkishStemSuffixCandidateGenerator.TAG_SEPARATOR_REGEX.split(tag)
            if stem_tags:
                if tag_sequences[0] in stem_tags or "+".join(tag_sequences[0:2]) in stem_tags:
                    res.append(tag_sequences)
            else:
                res.append(tag_sequences)
        return res

    def get_analysis_candidates(self, surface_word):
        candidate_analyzes = []
        candidate_roots, candidate_suffixes = self.get_stem_suffix_candidates(surface_word)
        for candidate_root, candidate_suffix in zip(candidate_roots, candidate_suffixes):
            if TurkishStemSuffixCandidateGenerator.NON_WORD_REGEX.match(candidate_root):
                stem_tags = ["Punc", "Num", "Noun+Time"]
            elif len(candidate_suffix) == 0 and candidate_root not in self.stem_dic:
                stem_tags = ["Noun", "Noun+Prop"]
            elif candidate_root not in self.stem_dic:
                if candidate_suffix.startswith("'"):
                    stem_tags = ["Noun+Prop", "Num"]
                else:
                    continue
            else:
                stem_tags = self.stem_dic[candidate_root]
            cur_candidate_analyzes = self._get_tags(candidate_suffix, stem_tags)
            if cur_candidate_analyzes and len(cur_candidate_analyzes) > 0:
                candidate_analyzes += [(candidate_root, candidate_suffix, cur_candidate_analysis)
                                       for cur_candidate_analysis in cur_candidate_analyzes]
        return candidate_analyzes


if __name__ == "__main__":
    candidate_generator = TurkishStemSuffixCandidateGenerator(case_sensitive=False)
    print(candidate_generator.get_analysis_candidates("MahalleYE"))

