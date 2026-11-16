# 方面抽取要处理口语同义词、繁体、空格、否定表达和按句极性
from app.services.stats_service import extract_aspects, match_aspect


def test_battery_synonyms_are_matched():
    assert extract_aspects("待机时间太短，一天要充三次电")["battery"] == -1
    assert extract_aspects("快充很快，续航也不错")["battery"] == 1
    assert extract_aspects("电池容量 5000 mAh，很耐用")["battery"] == 1


def test_traditional_and_spacing_are_normalized():
    assert extract_aspects("電池續航很差")["battery"] == -1
    assert extract_aspects("电 池 很 耐 用")["battery"] == 1


def test_negation_is_not_negative():
    # “不烫”不应被判成发热负面
    assert "heating" not in extract_aspects("用久了也不烫")
    # “电池没有问题”也不应判负
    assert extract_aspects("电池没有问题，续航稳定")["battery"] == 1


def test_long_negation_window():
    # “不会突然断电”“不仅续航好”里的否定词不应把整句判负
    assert extract_aspects("屏幕显示剩余电量很准确，不会突然断电")["battery"] == 1
    assert extract_aspects("不仅续航好，充电也快")["battery"] == 1


def test_polarity_is_judged_per_sentence():
    # 正面描述与负面描述在同一评论里，按“任一句负面即负面”的口径
    assert extract_aspects("电池很耐用。但是充电特别慢")["battery"] == -1
    # 负面词出现在别的句子里，不应污染电池方面
    assert extract_aspects("电池续航很好。就是物流太慢")["battery"] == 1


def test_question_matching_uses_same_lexicon():
    assert match_aspect("待机表现怎么样？") == "battery"
    assert match_aspect("掉电快吗") == "battery"
    assert match_aspect("做工怎么样") == "design"
