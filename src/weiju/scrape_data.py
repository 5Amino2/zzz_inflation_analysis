#!/usr/bin/env python3
"""
绝区零 危局强袭战 - 怪物数据爬取脚本
目标网站: https://zzz.nanoka.cc/boss/
功能: 批量爬取指定赛季ID范围的危局强袭战怪物数据（阶段一二三），输出为Excel表格
"""

import asyncio
import re
import pandas as pd
from playwright.async_api import async_playwright


# ==================== 配置区域 ====================

# 要爬取的赛季ID范围 (69001-69041)
SEASON_IDS = list(range(69001, 69042))

# 目标网站基础URL
BASE_URL = "https://zzz.nanoka.cc/boss/"

# 输出Excel文件路径
OUTPUT_PATH = "危局强袭战怪物数据_69001-69041.xlsx"

# 中文locale设置（确保页面显示中文内容）
LOCALE = "zh-CN"


# ==================== 页面爬取函数 ====================

async def fetch_page_text(season_id):
    """
    使用Playwright获取单个赛季页面的完整页面文本

    原理:
      - 网站使用SvelteKit框架，数据通过JavaScript动态渲染
      - 纯requests无法获取到怪物数据（JS未执行）
      - Playwright可以模拟浏览器完整渲染页面
    """
    url = f"{BASE_URL}{season_id}"

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(locale=LOCALE)
        page = await context.new_page()

        try:
            await page.goto(url, wait_until="networkidle", timeout=30000)
            await asyncio.sleep(3)
            text = await page.evaluate("() => document.body.innerText")
            await browser.close()
            return text

        except Exception as e:
            print(f"[错误] 赛季 {season_id} 获取失败: {e}")
            await browser.close()
            return None


async def fetch_all_seasons(season_ids, delay=1.5):
    """
    批量爬取多个赛季页面的数据

    参数:
        season_ids: 赛季ID列表
        delay: 请求间隔（秒），避免请求过快

    返回:
        dict: {赛季ID: 页面文本}
    """
    results = {}
    for season_id in season_ids:
        print(f"[爬取] 赛季 {season_id} ...")
        text = await fetch_page_text(season_id)
        if text:
            results[season_id] = text
        await asyncio.sleep(delay)
    return results


# ==================== 数据解析函数 ====================

def parse_boss_data(text, season_id):
    """
    从页面文本中解析危局强袭战的怪物数据

    解析逻辑:
      1. 以 "ID XXXXXXX" 作为锚点定位每个阶段
      2. 向前查找阶段标识（"阶段 X" 或怪物名称）
      3. 向后查找怪物名称（DEF行前面的一行）
      4. 提取生命值（低等级 / Lv.29 两档）

    参数:
        text: 页面完整文本
        season_id: 赛季ID

    返回:
        list: 怪物数据字典列表
    """
    results = []
    lines = [l.strip() for l in text.split("\n") if l.strip()]

    i = 0
    while i < len(lines):
        line = lines[i]

        # --- 检测怪物ID行 ---
        id_match = re.match(r"^ID\s+(\d+)$", line)
        if not id_match:
            i += 1
            continue

        monster_id = id_match.group(1)
        # 阶段号取ID最后一位（如 6900101 -> 1，6900102 -> 2）
        phase_num = int(monster_id[-1])

        # --- 向前查找阶段标识 ---
        phase_label = ""
        j = i - 1
        while j >= max(0, i - 8):
            prev = lines[j]
            # 跳过已知非标识行
            if prev in ["区域增益", "战斗房间", "", "可选增益"]:
                j -= 1
                continue
            if re.match(r"^ID\s+\d+$", prev):
                break
            if prev.startswith("\u00b7"):
                j -= 1
                continue
            # 匹配 "阶段 X" 或怪物名称
            if re.match(r"^阶段\s+[一二三123]$", prev) or len(prev) > 0:
                phase_label = prev
                break
            j -= 1

        # --- 向后查找怪物名称 ---
        monster_name = ""
        j = i + 1
        while j < min(len(lines), i + 20):
            if re.match(r"^DEF\s+[\d,]+", lines[j]):
                # DEF 行正上方的一行即为怪物名称
                if j > 0:
                    candidate = lines[j - 1]
                    if candidate not in ["弱点", "战斗房间 1", "战斗房间", "Lv.70"]:
                        monster_name = candidate
                break
            j += 1

        # --- 向后查找生命值 ---
        hp_low = ""   # 低等级生命值（第一列）
        hp_high = ""  # Lv.29 生命值（第二列）
        j = i + 1
        while j < min(len(lines), i + 30):
            if lines[j] == "生命值":
                if j + 1 < len(lines):
                    hp_low = lines[j + 1]
                if j + 2 < len(lines):
                    hp_high = lines[j + 2]
                break
            j += 1

        results.append({
            "赛季ID": season_id,
            "阶段": phase_num,
            "阶段标识": phase_label,
            "怪物ID": monster_id,
            "怪物名称": monster_name,
            "生命值_低等级": hp_low,
            "生命值_Lv29": hp_high,
        })

        i += 1

    return results


# ==================== 主函数 ====================

async def main():
    """主函数：爬取数据 -> 解析 -> 导出Excel"""

    print("=" * 60)
    print("绝区零 危局强袭战 - 怪物数据爬取")
    print("=" * 60)
    print(f"目标赛季: {SEASON_IDS[0]} ~ {SEASON_IDS[-1]} (共{len(SEASON_IDS)}个)")
    print(f"输出文件: {OUTPUT_PATH}")
    print()

    # ---- 步骤1: 批量爬取页面数据 ----
    print("[阶段1] 正在爬取页面数据...")
    all_texts = await fetch_all_seasons(SEASON_IDS)
    print(f"[完成] 成功获取 {len(all_texts)} 个赛季数据\n")

    # ---- 步骤2: 解析怪物数据 ----
    print("[阶段2] 正在解析怪物数据...")
    all_records = []
    for season_id in sorted(all_texts.keys()):
        records = parse_boss_data(all_texts[season_id], season_id)
        all_records.extend(records)
        print(f"  赛季 {season_id}: {len(records)} 条记录")

    print(f"[完成] 共解析 {len(all_records)} 条记录\n")

    # ---- 步骤3: 导出Excel ----
    print("[阶段3] 正在导出Excel...")

    column_order = [
        "赛季ID", "阶段", "阶段标识", "怪物ID", "怪物名称",
        "生命值_低等级", "生命值_Lv29"
    ]

    df = pd.DataFrame(all_records)
    df = df[column_order]
    df.to_excel(OUTPUT_PATH, index=False, sheet_name="危局强袭战怪物数据")

    print(f"[完成] 数据已保存: {OUTPUT_PATH}")
    print(f"  总行数: {len(df)} 条")
    print()
    print("=" * 60)
    print("爬取完成!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
