#!/usr/bin/env python3
"""
绝区零 式舆防卫战 - 剧变节点第五防线怪物数据爬取脚本
目标网站: https://zzz.nanoka.cc/shiyu/
功能: 批量爬取指定节点ID范围的剧变节点第五防线怪物数据，输出为Excel表格
"""

import asyncio
import re
import pandas as pd
from playwright.async_api import async_playwright


# ==================== 配置区域 ====================

# 要爬取的节点ID范围 (62038-62050)
NODE_IDS = list(range(62038, 62051))

# 目标网站基础URL
BASE_URL = "https://zzz.nanoka.cc/shiyu/"

# 输出Excel文件路径
OUTPUT_PATH = "剧变节点第五防线怪物数据_62038-62050.xlsx"

# 中文locale设置（确保页面显示中文内容）
LOCALE = "zh-CN"


# ==================== 页面爬取函数 ====================

async def fetch_page_text(node_id):
    """
    使用Playwright获取单个节点的完整页面文本
    
    原理:
      - 网站使用SvelteKit框架，数据通过JavaScript动态渲染
      - 纯requests无法获取到怪物数据（JS未执行）
      - Playwright可以模拟浏览器完整渲染页面
    """
    url = f"{BASE_URL}{node_id}"
    
    async with async_playwright() as p:
        # 启动无头浏览器
        browser = await p.chromium.launch(headless=True)
        
        # 创建中文浏览器上下文（确保页面显示中文）
        context = await browser.new_context(locale=LOCALE)
        page = await context.new_page()
        
        try:
            # 访问页面，等待网络空闲（所有JS数据加载完成）
            await page.goto(url, wait_until="networkidle", timeout=30000)
            
            # 额外等待3秒确保JavaScript渲染完成
            await asyncio.sleep(3)
            
            # 提取页面完整文本内容
            text = await page.evaluate("() => document.body.innerText")
            
            await browser.close()
            return text
            
        except Exception as e:
            print(f"[错误] 节点 {node_id} 获取失败: {e}")
            await browser.close()
            return None


async def fetch_all_nodes(node_ids, delay=1):
    """
    批量爬取多个节点的页面数据
    
    参数:
        node_ids: 节点ID列表
        delay: 节点间请求间隔（秒），避免请求过快
    
    返回:
        dict: {节点ID: 页面文本}
    """
    results = {}
    
    for node_id in node_ids:
        print(f"[爬取] 节点 {node_id} ...")
        text = await fetch_page_text(node_id)
        if text:
            results[node_id] = text
        await asyncio.sleep(delay)
    
    return results


# ==================== 数据解析函数 ====================

def parse_fifth_frontier(text, node_id):
    """
    从页面文本中解析第五防线的怪物数据
    
    解析逻辑:
      1. 定位"剧变节点第五防线"标记
      2. 在"剧变节点第四/六防线"处结束
      3. 逐行解析3个房间的数据
      4. 提取每个怪物的名称、HP、ATK、DEF、Stun、Anomaly、弱点、抗性
    
    参数:
        text: 页面完整文本
        node_id: 节点ID
    
    返回:
        list: 怪物数据字典列表
    """
    results = []
    
    # ===== 步骤1: 定位第五防线区域 =====
    start_idx = text.find("剧变节点第五防线")
    if start_idx == -1:
        print(f"[警告] 节点 {node_id}: 未找到第五防线数据")
        return results
    
    # 找到结束位置（下一个防线标记）
    end_markers = ["剧变节点第四防线", "剧变节点第六防线", "剧变节点第三防线"]
    end_idx = len(text)
    for marker in end_markers:
        idx = text.find(marker, start_idx + 10)
        if idx != -1 and idx < end_idx:
            end_idx = idx
    
    defense_text = text[start_idx:end_idx]
    lines = [l.strip() for l in defense_text.split("\n") if l.strip()]
    
    # ===== 步骤2: 逐行解析数据 =====
    current_room = None
    current_battle_room = None
    current_wave_count = None
    current_weakness_types = []
    current_zone_buff_name = ""
    current_zone_buff_desc = ""
    
    i = 0
    while i < len(lines):
        line = lines[i]
        
        # --- 检测房间标记 (房间一/二/三) ---
        room_match = re.match(r"^房间([一二三])$", line)
        if room_match:
            current_room = f"房间{room_match.group(1)}"
            current_battle_room = None
            current_wave_count = None
            current_weakness_types = []
            current_zone_buff_name = ""
            current_zone_buff_desc = ""
            i += 1
            continue
        
        # --- 检测区域增益名称 ---
        # 增益名称通常是2-5个汉字，后面跟着"·"开头的描述行
        if (not line.startswith("\u00b7")            # 不是描述行
            and len(line) <= 10                       # 短文本
            and i + 1 < len(lines)                    # 有下一行
            and lines[i + 1].startswith("\u00b7")):   # 下一行是描述
            
            # 排除已知的非增益名称
            non_buffs = {
                "房间一", "房间二", "房间三", "战斗房间", "Waves",
                "弱点", "抗性", "等级条件", "S:", "A:", "B:",
                "物理", "火属性", "电属性", "冰属性", "以太属性", "风属性",
                "火", "电", "冰", "以太", "风"
            }
            if line not in non_buffs and not re.match(r"^(HP|ATK|DEF|Stun|Anomaly|Lv\.|ID\s)", line):
                current_zone_buff_name = line
                i += 1
                continue
        
        # --- 收集增益描述（以"·"开头的描述行，排除通用规则描述） ---
        if line.startswith("\u00b7") and not any(k in line for k in ["本关内", "击败", "加成后", "造成伤害", "分数"]):
            desc_text = line[1:].strip()
            if current_zone_buff_desc:
                current_zone_buff_desc += " | " + desc_text
            else:
                current_zone_buff_desc = desc_text
            i += 1
            continue
        
        # --- 检测战斗房间号 ---
        battle_match = re.match(r"^战斗\s*房间\s+(\d+)$", line)
        if battle_match:
            current_battle_room = battle_match.group(1)
            i += 1
            continue
        
        # --- 检测波次数 ---
        wave_match = re.match(r"^Waves\s+(\d+)$", line)
        if wave_match:
            current_wave_count = wave_match.group(1)
            i += 1
            continue
        
        # --- 检测房间弱点属性 ---
        if line == "弱点":
            weakness_types = []
            j = i + 1
            valid_attrs = ["物理", "火", "火属性", "电", "电属性", "冰", "冰属性", "以太", "以太属性"]
            while j < len(lines) and lines[j] in valid_attrs:
                w = lines[j].replace("属性", "").strip()
                if w not in weakness_types:
                    weakness_types.append(w)
                j += 1
            if weakness_types:
                current_weakness_types = weakness_types
            i = j
            continue
        
        # --- 检测怪物数据 ---
        # 怪物格式: 名称行 + HP/ATK/DEF行 + Stun/Anomaly行 + 弱点/抗性行
        if i + 1 < len(lines):
            next_line = lines[i + 1]
            hp_match = re.match(
                r"^HP\s+([\d,]+)\s+\u00b7\s+ATK\s+([\d,]+)\s+\u00b7\s+DEF\s+([\d,]+)$",
                next_line
            )
            
            if hp_match:
                monster_name = line
                hp = hp_match.group(1)
                atk = hp_match.group(2)
                def_val = hp_match.group(3)
                
                # 提取Stun和Anomaly
                stun_val = ""
                anomaly_val = ""
                if i + 2 < len(lines):
                    stun_match = re.match(
                        r"^Stun\s+([\d,]+)\s+\u00b7\s+Anomaly\s+([\d,]+)$",
                        lines[i + 2]
                    )
                    if stun_match:
                        stun_val = stun_match.group(1)
                        anomaly_val = stun_match.group(2)
                
                # 提取怪物弱点
                weaknesses = []
                j = i + 3
                while j < len(lines) and j < i + 6:
                    if lines[j] == "弱点":
                        j += 1
                        valid_attrs = ["物理", "火", "火属性", "电", "电属性", "冰", "冰属性", "以太", "以太属性"]
                        while j < len(lines) and lines[j] in valid_attrs:
                            w = lines[j].replace("属性", "").strip()
                            if w not in weaknesses:
                                weaknesses.append(w)
                            j += 1
                        break
                    j += 1
                
                # 提取怪物抗性
                resistances = []
                j = i + 3
                while j < len(lines) and j < i + 7:
                    if lines[j] == "抗性":
                        j += 1
                        valid_attrs = ["物理", "火", "火属性", "电", "电属性", "冰", "冰属性", "以太", "以太属性", "风"]
                        while j < len(lines) and lines[j] in valid_attrs:
                            r = lines[j].replace("属性", "").strip()
                            if r not in resistances:
                                resistances.append(r)
                            j += 1
                        break
                    j += 1
                
                # 保存怪物数据
                results.append({
                    "节点ID": node_id,
                    "防线": "第五防线",
                    "房间": current_room or "",
                    "区域增益名称": current_zone_buff_name,
                    "区域增益描述": current_zone_buff_desc,
                    "战斗房间": current_battle_room or "",
                    "波次数": current_wave_count or "",
                    "房间弱点属性": ", ".join(current_weakness_types) if current_weakness_types else "",
                    "怪物名称": monster_name,
                    "HP": hp,
                    "ATK": atk,
                    "DEF": def_val,
                    "Stun": stun_val,
                    "Anomaly": anomaly_val,
                    "怪物弱点": ", ".join(weaknesses) if weaknesses else "",
                    "怪物抗性": ", ".join(resistances) if resistances else ""
                })
                
                i += 3  # 跳过已处理的行
                continue
        
        i += 1
    
    return results


# ==================== 主函数 ====================

async def main():
    """主函数：爬取数据 -> 解析 -> 导出Excel"""
    
    print("=" * 60)
    print("绝区零 式舆防卫战 - 剧变节点第五防线怪物数据爬取")
    print("=" * 60)
    print(f"目标节点: {NODE_IDS[0]} ~ {NODE_IDS[-1]} (共{len(NODE_IDS)}个)")
    print(f"输出文件: {OUTPUT_PATH}")
    print()
    
    # ---- 步骤1: 批量爬取页面数据 ----
    print("[阶段1] 正在爬取页面数据...")
    all_texts = await fetch_all_nodes(NODE_IDS)
    print(f"[完成] 成功获取 {len(all_texts)} 个节点数据\n")
    
    # ---- 步骤2: 解析怪物数据 ----
    print("[阶段2] 正在解析怪物数据...")
    all_monsters = []
    for node_id in sorted(all_texts.keys()):
        monsters = parse_fifth_frontier(all_texts[node_id], node_id)
        all_monsters.extend(monsters)
        print(f"  节点 {node_id}: {len(monsters)} 只怪物")
    
    print(f"[完成] 共解析 {len(all_monsters)} 条怪物记录\n")
    
    # ---- 步骤3: 导出Excel ----
    print("[阶段3] 正在导出Excel...")
    
    column_order = [
        "节点ID", "防线", "房间", "区域增益名称", "区域增益描述",
        "战斗房间", "波次数", "房间弱点属性",
        "怪物名称", "HP", "ATK", "DEF", "Stun", "Anomaly",
        "怪物弱点", "怪物抗性"
    ]
    
    df = pd.DataFrame(all_monsters)
    df = df[column_order]
    df.to_excel(OUTPUT_PATH, index=False, sheet_name="怪物数据")
    
    print(f"[完成] 数据已保存: {OUTPUT_PATH}")
    print(f"  总行数: {len(df)} 条")
    print()
    print("=" * 60)
    print("爬取完成!")
    print("=" * 60)


if __name__ == "__main__":
    # 运行异步主函数
    asyncio.run(main())
