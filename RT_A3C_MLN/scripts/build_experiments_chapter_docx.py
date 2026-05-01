from __future__ import annotations

from pathlib import Path
import os

import pandas as pd
from docx import Document
from docx.enum.section import WD_ORIENT, WD_SECTION
from docx.enum.table import WD_ALIGN_VERTICAL
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Inches, Pt, RGBColor


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = Path(os.environ.get("PAPER_ARTIFACTS_DIR", ROOT / "paper_artifacts"))
FIGURES = ARTIFACTS / "figures"
TABLES = ARTIFACTS / "tables"
OUT = ARTIFACTS / "RT_A3C_MLN_实验章节报告.docx"
STRONG = ARTIFACTS / "strong_rule_drift"


ACCENT = "1F4E79"
ACCENT_LIGHT = "EAF3F8"
HEADER_FILL = "D9EAF7"
GRAY_FILL = "F4F6F8"


def set_east_asian_font(run, font="宋体", size=None, bold=None, color=None):
    run.font.name = font
    run._element.rPr.rFonts.set(qn("w:eastAsia"), font)
    if size is not None:
        run.font.size = Pt(size)
    if bold is not None:
        run.bold = bold
    if color is not None:
        run.font.color.rgb = RGBColor.from_string(color)


def set_paragraph_font(paragraph, font="宋体", size=10.5, color=None):
    for run in paragraph.runs:
        set_east_asian_font(run, font=font, size=size, color=color)


def shade_cell(cell, fill):
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = tc_pr.find(qn("w:shd"))
    if shd is None:
        shd = OxmlElement("w:shd")
        tc_pr.append(shd)
    shd.set(qn("w:fill"), fill)


def set_cell_margins(cell, top=90, start=90, bottom=90, end=90):
    tc = cell._tc
    tc_pr = tc.get_or_add_tcPr()
    tc_mar = tc_pr.first_child_found_in("w:tcMar")
    if tc_mar is None:
        tc_mar = OxmlElement("w:tcMar")
        tc_pr.append(tc_mar)
    for m, v in {"top": top, "start": start, "bottom": bottom, "end": end}.items():
        node = tc_mar.find(qn(f"w:{m}"))
        if node is None:
            node = OxmlElement(f"w:{m}")
            tc_mar.append(node)
        node.set(qn("w:w"), str(v))
        node.set(qn("w:type"), "dxa")


def set_table_borders(table, color="B7C7D6", size="4"):
    tbl = table._tbl
    tbl_pr = tbl.tblPr
    borders = tbl_pr.first_child_found_in("w:tblBorders")
    if borders is None:
        borders = OxmlElement("w:tblBorders")
        tbl_pr.append(borders)
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        tag = f"w:{edge}"
        element = borders.find(qn(tag))
        if element is None:
            element = OxmlElement(tag)
            borders.append(element)
        element.set(qn("w:val"), "single")
        element.set(qn("w:sz"), size)
        element.set(qn("w:space"), "0")
        element.set(qn("w:color"), color)


def set_repeat_table_header(row):
    tr_pr = row._tr.get_or_add_trPr()
    tbl_header = OxmlElement("w:tblHeader")
    tbl_header.set(qn("w:val"), "true")
    tr_pr.append(tbl_header)


def add_heading(doc, text, level=1):
    p = doc.add_paragraph()
    p.style = f"Heading {level}"
    run = p.add_run(text)
    set_east_asian_font(run, font="微软雅黑", size={1: 15, 2: 13, 3: 11.5}.get(level, 11), bold=True, color=ACCENT)
    p.paragraph_format.space_before = Pt(10 if level == 1 else 7)
    p.paragraph_format.space_after = Pt(5)
    return p


def add_body(doc, text, first_line=True):
    p = doc.add_paragraph()
    p.style = "Normal"
    p.paragraph_format.line_spacing = 1.25
    p.paragraph_format.space_after = Pt(5)
    if first_line:
        p.paragraph_format.first_line_indent = Pt(21)
    run = p.add_run(text)
    set_east_asian_font(run, font="宋体", size=10.5)
    return p


def add_bullets(doc, items):
    for item in items:
        p = doc.add_paragraph(style="List Bullet")
        p.paragraph_format.left_indent = Pt(18)
        p.paragraph_format.space_after = Pt(3)
        run = p.add_run(item)
        set_east_asian_font(run, font="宋体", size=10.2)


def add_caption(doc, text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(3)
    p.paragraph_format.space_after = Pt(7)
    run = p.add_run(text)
    set_east_asian_font(run, font="宋体", size=9.2, bold=True, color="333333")
    return p


def add_note_box(doc, title, body):
    table = doc.add_table(rows=1, cols=1)
    table.alignment = WD_ALIGN_PARAGRAPH.CENTER
    set_table_borders(table, color="BFD3E3", size="6")
    cell = table.cell(0, 0)
    shade_cell(cell, ACCENT_LIGHT)
    set_cell_margins(cell, top=160, bottom=160, start=180, end=180)
    p = cell.paragraphs[0]
    p.paragraph_format.space_after = Pt(4)
    r = p.add_run(title)
    set_east_asian_font(r, font="微软雅黑", size=10.5, bold=True, color=ACCENT)
    p2 = cell.add_paragraph()
    p2.paragraph_format.line_spacing = 1.18
    r2 = p2.add_run(body)
    set_east_asian_font(r2, font="宋体", size=10)
    doc.add_paragraph()


def add_picture(doc, image_path, caption, width_inches=6.3):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run()
    run.add_picture(str(image_path), width=Inches(width_inches))
    add_caption(doc, caption)


def format_table(table, font_size=8.0, header_size=8.0):
    table.style = "Table Grid"
    table.autofit = True
    set_table_borders(table)
    for i, row in enumerate(table.rows):
        for cell in row.cells:
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            set_cell_margins(cell, top=80, bottom=80, start=65, end=65)
            if i == 0:
                shade_cell(cell, HEADER_FILL)
            for p in cell.paragraphs:
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER if i == 0 else WD_ALIGN_PARAGRAPH.CENTER
                p.paragraph_format.line_spacing = 1.05
                p.paragraph_format.space_after = Pt(0)
                for r in p.runs:
                    set_east_asian_font(
                        r,
                        font="宋体",
                        size=header_size if i == 0 else font_size,
                        bold=True if i == 0 else None,
                    )
    set_repeat_table_header(table.rows[0])


def add_dataframe_table(doc, df, caption, font_size=7.2, header_size=7.0):
    add_caption(doc, caption)
    table = doc.add_table(rows=1, cols=len(df.columns))
    for j, col in enumerate(df.columns):
        table.cell(0, j).text = str(col)
    for _, row in df.iterrows():
        cells = table.add_row().cells
        for j, col in enumerate(df.columns):
            cells[j].text = str(row[col])
    format_table(table, font_size=font_size, header_size=header_size)
    doc.add_paragraph()
    return table


def set_section_portrait(section):
    section.orientation = WD_ORIENT.PORTRAIT
    section.page_width = Cm(21)
    section.page_height = Cm(29.7)
    section.top_margin = Cm(2.1)
    section.bottom_margin = Cm(2.0)
    section.left_margin = Cm(2.2)
    section.right_margin = Cm(2.0)


def set_section_landscape(section):
    section.orientation = WD_ORIENT.LANDSCAPE
    section.page_width = Cm(29.7)
    section.page_height = Cm(21)
    section.top_margin = Cm(1.6)
    section.bottom_margin = Cm(1.5)
    section.left_margin = Cm(1.4)
    section.right_margin = Cm(1.4)


def add_landscape_section(doc):
    section = doc.add_section(WD_SECTION.NEW_PAGE)
    set_section_landscape(section)
    return section


def add_portrait_section(doc):
    section = doc.add_section(WD_SECTION.NEW_PAGE)
    set_section_portrait(section)
    return section


def prepare_main_table():
    df = pd.read_csv(TABLES / "Table1_Main_Results.csv")
    rename = {
        "Method": "方法",
        "Streaming update protocol": "更新协议",
        "Adaptive rules": "规则自适应",
        "F1": "F1",
        "AUC": "AUC",
        "Log-loss": "Log-loss",
        "FG update ms": "前台更新(ms)",
        "Blocking cycle ms": "阻塞周期(ms)",
        "10ms success %": "10ms成功率(%)",
        "Struct update %": "结构更新(%)",
        "Rule +/- per chunk": "规则增/删",
        "Update slowdown vs ours": "更新慢于本文",
    }
    df = df.rename(columns=rename)
    replacements = {
        "RT-A3C-MLN (ours)": "RT-A3C-MLN",
        "Rolling BeamSearch MLN": "Rolling BeamSearch",
        "Static MaxEnt reference": "Static MaxEnt",
    }
    df["方法"] = df["方法"].replace(replacements)
    df["更新协议"] = df["更新协议"].replace(
        {
            "async A3C rule/weight adaptation": "异步A3C规则/权重更新",
            "rolling global MLN refit": "滚动全局重训",
            "online fixed-rule reference": "固定规则在线权重",
            "static non-adaptive reference": "静态非自适应",
        }
    )
    df["规则自适应"] = df["规则自适应"].replace({"Yes": "是", "No": "否"})
    return df


def prepare_ablation_table():
    df = pd.read_csv(TABLES / "Table2_Ablation.csv")
    rename = {
        "Variant": "版本",
        "Controlled change": "控制变量",
        "F1": "F1",
        "AUC": "AUC",
        "Log-loss": "Log-loss",
        "FG update ms": "前台更新(ms)",
        "Blocking cycle ms": "阻塞周期(ms)",
        "Struct update %": "结构更新(%)",
        "Rule +/chunk": "新增规则/chunk",
        "Rule -/chunk": "删除规则/chunk",
        "10ms success %": "10ms成功率(%)",
    }
    df = df.rename(columns=rename)
    df["版本"] = df["版本"].replace(
        {
            "Full RT-A3C-MLN": "Full RT-A3C",
            "No prior-shift update": "No prior-shift",
            "No structure refresh": "No structure",
            "Online only, no A3C": "Online only",
        }
    )
    df["控制变量"] = df["控制变量"].replace(
        {
            "complete realtime foreground update": "完整实时更新",
            "removes parallel actor exploration": "去除多Actor并行",
            "removes local stream-prior adaptation": "去除局部先验漂移",
            "disables background rule-structure refresh": "关闭后台结构刷新",
            "keeps online weights, removes A3C structure learning": "仅保留在线权重",
        }
    )
    return df


def add_metric_definitions(doc):
    items = [
        "F1：精确率与召回率的调和平均，用于衡量分类性能，数值越高越好。",
        "AUC：ROC 曲线下面积，用于衡量模型排序能力，数值越高表示区分正负类的能力越强。",
        "Log-loss：对数损失，用于评价概率输出的校准质量，数值越低表示概率推理越可靠。",
        "FG update ms：前台更新耗时，即新数据反馈到达后必须阻塞完成的权重或局部统计更新耗时。",
        "Blocking cycle ms：推理耗时与前台更新耗时之和，是本文判断实时响应能力的核心指标。",
        "10ms success %：阻塞周期不超过 10 ms 的数据流 chunk 比例，用于衡量实时截止约束的满足程度。",
        "Struct update % 与 Rule +/-：分别表示后台规则结构刷新被接受的比例，以及每个 chunk 平均新增/删除规则数，用于证明规则结构是否在数据流阶段发生自适应调整。",
    ]
    add_bullets(doc, items)


def build():
    main_df = prepare_main_table()
    ablation_df = prepare_ablation_table()

    doc = Document()
    set_section_portrait(doc.sections[0])

    styles = doc.styles
    styles["Normal"].font.name = "宋体"
    styles["Normal"]._element.rPr.rFonts.set(qn("w:eastAsia"), "宋体")
    styles["Normal"].font.size = Pt(10.5)
    for style_name in ("Heading 1", "Heading 2", "Heading 3"):
        styles[style_name].font.name = "微软雅黑"
        styles[style_name]._element.rPr.rFonts.set(qn("w:eastAsia"), "微软雅黑")

    title = doc.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title.paragraph_format.space_before = Pt(18)
    title.paragraph_format.space_after = Pt(8)
    r = title.add_run("第4章 实验与结果分析")
    set_east_asian_font(r, font="微软雅黑", size=18, bold=True, color=ACCENT)

    subtitle = doc.add_paragraph()
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = subtitle.add_run("基于异步优势 Actor-Critic 的实时 MLN 规则学习与量化")
    set_east_asian_font(r, font="微软雅黑", size=11, color="555555")

    add_note_box(
        doc,
        "章节主线",
        "本章实验围绕一个核心问题展开：在连续到达的数据流场景下，本文提出的 RT-A3C-MLN 是否能够在保持毫秒级实时响应的同时，对 MLN 规则结构和规则权重进行自适应调整，并输出稳定的概率推理结果。",
    )

    add_heading(doc, "4.1 实验目的与验证思路", 1)
    add_body(
        doc,
        "传统 MLN 结构学习通常依赖全局数据上的对数似然最大化，需要在候选规则空间中反复搜索和重估权重。当数据分布随时间变化时，这类全局重训过程会产生明显更新滞后，难以直接支撑实时推理服务。本文方法将异步优势 Actor-Critic 引入 MLN 规则学习过程，通过多个 Actor 在概念格环境中并行探索规则添加、删除、修改和权重调整动作，并将梯度异步反馈至全局 Actor-Critic 网络。",
    )
    add_body(
        doc,
        "实验设计不把目标限定为单一准确率最优，而是同时考察预测性能、概率质量、更新延迟、实时截止成功率和规则结构变化。只要模型能够在不同数据流输入下保持低阻塞时间，并出现可验证的规则新增、删除或结构刷新，就能为“实时响应的概率知识网络”这一核心论点提供证据。",
    )

    add_picture(
        doc,
        FIGURES / "Fig1_Method_Framework.png",
        "图4-1 基于异步优势 Actor-Critic 的实时 MLN 规则学习框架",
        width_inches=6.25,
    )
    add_body(
        doc,
        "图4-1 展示了本文方法的运行机制。每个概念格被视为一个独立学习环境，Actor 根据当前 MLN 规则集合和权重状态选择动作，动作包括规则添加、删除、修改以及权重调整；Critic 估计状态价值并通过优势函数降低策略梯度方差。多个 Actor 的异步探索结果汇入全局网络，使模型能够在不同规则路径上并行学习。前台路径负责快速推理和轻量更新，后台路径负责被验证窗口接受的规则结构刷新，因此实时响应与结构自适应可以同时存在。",
    )

    add_heading(doc, "4.2 实验设置", 1)
    add_body(
        doc,
        "实验使用 mushroom、adult、bank 和 spambase 四个二分类数据集构造数据流场景。每个数据集保留 70% 样本作为流式阶段，并划分为 8 个连续 chunk，每个 chunk 最多包含 1200 个样本。为了模拟数据流中的分布漂移，实验记录每个 chunk 的正类比例变化，并在每次 chunk 到达后执行推理、反馈吸收和模型更新。",
    )
    add_body(
        doc,
        "规则候选生成阶段设置最大候选规则数为 220，数值特征划分为 5 个区间，最小支持度为 0.015。RT-A3C-MLN 使用 4 个 Actor 并行探索，n-step 返回步数为 5，初始 A3C 训练 episode 为 5，后台结构刷新间隔为 4 个 chunk，刷新 episode 为 2。前台更新主要承担局部先验漂移调整、在线权重更新和校准，后台结构刷新用于产生被验证窗口接受的规则结构变化。",
    )
    add_heading(doc, "4.3 对比方法与评价指标", 1)
    add_body(
        doc,
        "对比方法分为两类。第一类是具有结构学习或滚动重训性质的 MLN 基线，包括 Rolling L1 MLN、Rolling MaxEnt MLN、Rolling Boosted MLN 和 Rolling BeamSearch MLN。它们能够重新拟合规则或权重，但需要在数据流到达后执行较重的全局更新。第二类是轻量参考方法，包括 OnlineWeight MLN 和 Static MaxEnt reference。前者只在固定规则结构上更新权重，后者不随数据流更新，因此二者不能证明规则结构自适应，只用于说明固定结构或静态模型的速度上界。",
    )
    add_metric_definitions(doc)

    add_landscape_section(doc)
    add_heading(doc, "4.4 主实验结果", 1)
    add_dataframe_table(doc, main_df, "表4-1 主结果：预测性能、实时响应和规则结构自适应能力对比", font_size=6.8, header_size=6.7)
    add_body(
        doc,
        "表4-1 表明，RT-A3C-MLN 的平均前台更新时间为 4.839 ms，平均阻塞周期为 7.201 ms，10 ms 截止成功率达到 100%。相较于滚动重训类 MLN 基线，本文方法显著降低了数据流反馈后的阻塞成本：Rolling L1、Rolling MaxEnt、Rolling Boosted 和 Rolling BeamSearch 的前台更新耗时分别约为本文方法的 2.55 倍、3.52 倍、58.68 倍和 360.76 倍。该结果说明，异步 A3C 与前后台解耦更新能够把 MLN 的结构学习代价压缩到实时推理可以承受的范围内。",
    )
    add_body(
        doc,
        "从预测质量看，RT-A3C-MLN 的 F1 为 0.678，AUC 为 0.931，处于滚动 MLN 基线的竞争范围内。本文方法并不是以牺牲全部预测性能来换取速度，而是在保持可用分类性能和较好概率输出的同时，获得稳定的实时响应。OnlineWeight MLN 和 Static MaxEnt reference 虽然在速度或 F1 上看起来较好，但它们的规则结构不随数据流变化，Rule +/- 为 0，Struct update 也为 0，因此只能作为固定结构参考，不能替代本文要论证的规则结构自适应能力。",
    )

    add_portrait_section(doc)
    add_heading(doc, "4.5 实时性分析", 1)
    add_picture(
        doc,
        FIGURES / "Fig2_Realtime_Latency_Breakdown.png",
        "图4-2 前台更新时间与阻塞周期对比",
        width_inches=6.35,
    )
    add_body(
        doc,
        "图4-2 进一步展示了各方法的前台更新和阻塞周期差异。横向对比可以看到，RT-A3C-MLN 的阻塞周期位于 10 ms 实时阈值以内，而需要滚动重训或重新搜索规则结构的基线方法明显超出该阈值。该图的关键证据不是“本文方法绝对最快”，而是“在保留规则结构自适应的前提下仍然足够快”。OnlineWeight 和 Static MaxEnt 的速度优势来自固定结构或不更新模型，不具备规则添加、删除和结构刷新能力，因此不构成对本文核心论点的反证。",
    )

    add_picture(
        doc,
        FIGURES / "Fig3_Dataset_Deadline_Heatmap.png",
        "图4-3 不同数据集上的阻塞周期与 10 ms 截止成功率",
        width_inches=6.35,
    )
    doc.add_page_break()
    add_body(
        doc,
        "图4-3 从数据集维度验证实时性鲁棒性。左侧子图给出每个方法在不同数据集上的平均阻塞周期，右侧子图给出 10 ms 截止成功率。RT-A3C-MLN 在四个数据集上均保持低于 10 ms 的平均阻塞周期，并在所有 chunk 上满足实时截止要求。这说明本文方法的速度优势不是由某一个特定数据集造成的，而是在不同数据规模、特征类型和类别分布下均能保持稳定。",
    )

    add_heading(doc, "4.6 数据流漂移下的自适应分析", 1)
    add_picture(
        doc,
        FIGURES / "Fig4_Stream_Drift_Response.png",
        "图4-4 数据流漂移下的 F1 变化与阻塞周期变化",
        width_inches=6.35,
    )
    add_body(
        doc,
        "图4-4 关注动态数据流过程。上方子图的灰色区域和虚线表示不同 chunk 的平均正类比例变化，说明实验中存在明显类别先验漂移；彩色曲线表示各方法在连续 chunk 上的 F1。下方子图展示每个 chunk 的阻塞周期，并以红色虚线标出 10 ms 实时截止线。RT-A3C-MLN 在数据分布变化时保持 F1 的相对稳定，同时其阻塞周期始终处于 10 ms 以下，说明模型能够在吸收局部反馈后继续实时提供推理结果。",
    )
    add_body(
        doc,
        "这一结果直接对应本文的技术可行性论证：MLN 规则和权重不再完全依赖离线全局重训，而是可以通过前台快速更新与后台异步结构刷新共同响应数据流变化。相比之下，Rolling MaxEnt、Boosted 和 BeamSearch 虽然也能通过重训适应数据，但其更新代价过高，难以满足实时服务场景。",
    )

    add_landscape_section(doc)
    add_heading(doc, "4.7 消融实验", 1)
    add_dataframe_table(doc, ablation_df, "表4-2 消融实验：异步 Actor、局部先验适应和后台结构刷新的作用", font_size=7.3, header_size=7.0)
    add_body(
        doc,
        "表4-2 对 RT-A3C-MLN 的关键组件进行消融。Full RT-A3C 的阻塞周期为 6.737 ms，10 ms 成功率为 100%，同时结构更新比例为 18.8%，每个 chunk 平均新增和删除规则均约为 0.94。这说明完整模型不仅能够实时运行，而且确实在数据流阶段发生了规则结构变化。",
    )
    add_body(
        doc,
        "Single actor 去除了多 Actor 并行探索，F1 与完整模型接近，但规则变化数量略低，说明并行探索主要贡献在于增加规则空间覆盖和提升结构更新机会。No prior-shift update 的 Log-loss 从 0.364 增加到 0.387，说明局部先验漂移更新有助于概率输出校准。No structure refresh 的 F1 较高，但结构更新率和规则增删均为 0，说明其表现来自固定结构上的权重适应，不能证明 MLN 规则结构能够随数据流调整。Online only 速度最快且 F1 较高，但 AUC 降至 0.870，Log-loss 达到 1.227，说明固定规则在线权重更新的概率质量明显不足。",
    )

    add_portrait_section(doc)
    add_picture(
        doc,
        FIGURES / "Fig5_Ablation_Summary.png",
        "图4-5 消融实验中的预测性能、实时响应和规则结构变化",
        width_inches=6.35,
    )
    add_body(
        doc,
        "图4-5 将消融结果可视化为三个子图。左侧子图显示 F1 与 Log-loss 的关系，表明只看 F1 容易高估固定结构在线方法；中间子图显示所有 RT-A3C 相关版本均能把阻塞周期控制在 10 ms 以下；右侧子图显示完整模型、Single actor 和 No prior-shift 仍存在规则新增、删除和结构刷新，而 No structure 与 Online only 的规则变化为 0。该图证明规则结构自适应并非由在线权重更新自然产生，而是来自本文设计的 A3C 结构学习与后台结构刷新机制。",
    )

    add_heading(doc, "4.8 对论文核心论点的证据归纳", 1)
    add_body(
        doc,
        "综合上述实验，可以得到三点结论。第一，RT-A3C-MLN 相比滚动重训类 MLN 基线显著降低了数据流更新阻塞时间，平均阻塞周期保持在 10 ms 以内，支持“实时响应”的技术可行性。第二，结构更新比例和规则增删指标显示，完整模型在数据流阶段确实发生了规则结构变化，而不是仅在固定规则集合上调节权重。第三，Log-loss 与消融结果说明，A3C 结构学习、局部先验漂移更新和在线权重更新共同维持了概率推理质量。",
    )
    add_body(
        doc,
        "因此，本章实验支持本文主要论点：通过异步优势 Actor-Critic 的并行探索与异步更新机制，MLN 规则可以在不同数据流输入下进行自适应调整，并在毫秒级阻塞周期内持续输出概率推理结果。需要强调的是，本文方法的优势不是单纯追求最高 F1，而是在结构自适应、概率质量和实时响应之间取得平衡，从而形成面向数据流场景的实时概率知识网络。",
    )

    add_note_box(
        doc,
        "论文写作建议",
        "在正式论文中，OnlineWeight MLN 与 Static MaxEnt reference 应明确称为“固定结构参考模型”或“非自适应参考模型”。它们可用于说明固定结构方法的速度上界，但不应作为完整规则结构自适应方法与本文方法直接比较。主结论应围绕“结构可变 + 前台低阻塞 + 概率输出稳定”展开。",
    )

    doc.save(OUT)
    print(OUT)


if __name__ == "__main__":
    build()
