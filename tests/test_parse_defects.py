# 解析器要能容错，模型输出格式不对时接口不能出错
from app.services.agent_service import _parse_defects, _parse_merits, _parse_overall


def test_plain_json():
    raw = '{"defects": [{"name": "续航衰减", "evidence": "半年后续航明显下降"}]}'
    defects = _parse_defects(raw)
    assert defects == [{"name": "续航衰减", "evidence": "半年后续航明显下降"}]


def test_code_fence_and_prose():
    raw = '好的，结果如下：\n```json\n{"defects": [{"name": "充电慢", "evidence": "充电要两小时"}]}\n```\n以上。'
    defects = _parse_defects(raw)
    assert len(defects) == 1
    assert defects[0]["name"] == "充电慢"


def test_invalid_json_returns_empty():
    assert _parse_defects("模型这次没有按格式输出") == []
    assert _parse_defects("{not json}") == []


def test_missing_and_non_dict_items():
    raw = '{"defects": ["脏数据", {"evidence": "只有证据"}, {"name": "发热严重"}]}'
    defects = _parse_defects(raw)
    assert defects == [{"name": "发热严重", "evidence": ""}]


def test_truncate_to_ten():
    items = ",".join(f'{{"name": "缺点{i}"}}' for i in range(15))
    defects = _parse_defects('{"defects": [' + items + "]}")
    assert len(defects) == 10


def test_parse_merits_uses_same_item_shape():
    raw = '{"merits": [{"name": "续航稳定", "evidence": "一天一充够用"}]}'
    assert _parse_merits(raw) == [{"name": "续航稳定", "evidence": "一天一充够用"}]


def test_parse_overall_keeps_summary_and_lists():
    raw = (
        '{"summary": "整体不错", "merits": [{"name": "外观"}], '
        '"defects": [{"name": "续航"}], "extra": 1}'
    )
    data = _parse_overall(raw)
    assert data["summary"] == "整体不错"
    assert [item["name"] for item in data["merits"]] == ["外观"]
    assert [item["name"] for item in data["defects"]] == ["续航"]


def test_parse_overall_invalid_returns_empty():
    assert _parse_overall("模型这次没有按格式输出") == {}
    assert _parse_overall('{"merits": [], "defects": []}') == {}
