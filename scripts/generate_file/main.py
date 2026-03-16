import os
import random
import json
from faker import Faker
from docx import Document
from openpyxl import Workbook
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from pptx import Presentation
import pandas as pd
from tqdm import tqdm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont


fake = Faker("zh_CN")

ROOT = "data"

folders = [
"01_公司治理",
"02_行业研究",
"03_客户与项目",
"04_基金产品",
"05_风险管理",
"06_会议纪要",
"07_投资数据",
"08_市场数据",
"09_内部培训",
"10_内部邮件",
"11_投资备忘录"
]

for f in folders:
    os.makedirs(os.path.join(ROOT,f),exist_ok=True)

clients = [
"星云科技集团",
"东海新能源股份",
"华联消费集团",
"北辰医疗科技",
"天穹半导体",
"远航智能制造",
"蓝海能源科技"
]

funds = [
"HC Alpha Growth Fund",
"HC Global Macro Fund",
"HC Quant Arbitrage Fund",
"HC Energy Transition Fund",
"HC AI Innovation Fund"
]

industries = [
"人工智能",
"新能源",
"医疗科技",
"消费行业",
"半导体",
"机器人",
"云计算"
]

departments = [
"投资策略部",
"量化研究部",
"风险管理部",
"资产配置委员会",
"投资委员会"
]

sections = [
"行业概述",
"市场规模分析",
"产业链结构",
"竞争格局",
"重点企业研究",
"投资逻辑",
"风险因素",
"未来趋势"
]

def long_paragraph():

    company=random.choice(clients)
    fund=random.choice(funds)
    industry=random.choice(industries)
    dept=random.choice(departments)

    text=f"""
{company}目前是{industry}行业的重要参与者之一。根据华辰资本{dept}的内部研究，
该公司在过去五年内实现了稳定的收入增长，并在核心技术研发方面持续投入。
华辰资本旗下的 {fund} 在2024年至2025年期间对该企业进行了多轮调研，
包括管理层访谈、产业链走访以及财务模型测算。

从行业角度看，{industry}行业正处于技术创新和资本投入双重驱动的阶段。
根据内部研究报告《{industry}行业深度研究报告》，预计未来五年市场规模
将保持约15%至20%的复合增长率。

投资委员会在最近的会议纪要中指出，{industry}行业的投资机会主要集中在
技术平台型企业以及拥有规模优势的产业链龙头企业。
"""

    return text


def generate_long_text(paragraphs=120):

    text=""
    for _ in range(paragraphs):
        text+=long_paragraph()+"\n\n"

    return text


# Markdown
def generate_md(path):

    with open(path,"w",encoding="utf8") as f:

        f.write("# 行业深度研究报告\n\n")

        for i in range(1,6):

            f.write(f"## {i} 行业分析章节\n\n")

            for j in range(1,4):

                f.write(f"### {i}.{j} 子章节\n\n")

                for _ in range(6):
                    f.write(long_paragraph()+"\n\n")

                f.write("| 公司 | 行业 | 市值 | 收入增长 |\n")
                f.write("|----|----|----|----|\n")

                for _ in range(6):

                    f.write(
                        f"| {random.choice(clients)} | {random.choice(industries)} | {random.randint(100,800)}亿 | {random.randint(5,30)}% |\n"
                    )

                f.write("\n")


# DOCX
def generate_docx(path):

    doc=Document()

    doc.add_heading("行业投资研究报告",0)

    for i in range(1,6):

        doc.add_heading(f"{i} 行业章节",level=1)

        for j in range(1,4):

            doc.add_heading(f"{i}.{j} 子章节",level=2)

            for _ in range(6):
                doc.add_paragraph(long_paragraph())

            table=doc.add_table(rows=1,cols=4)

            hdr=table.rows[0].cells
            hdr[0].text="公司"
            hdr[1].text="行业"
            hdr[2].text="市值"
            hdr[3].text="增长率"

            for _ in range(5):

                row=table.add_row().cells
                row[0].text=random.choice(clients)
                row[1].text=random.choice(industries)
                row[2].text=str(random.randint(100,600))+"亿"
                row[3].text=str(random.randint(5,30))+"%"

    doc.save(path)
# PDF
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.units import cm
from reportlab.lib import colors

FONT_PATH = "fonts/NotoSansCJKsc-VF.ttf"

pdfmetrics.registerFont(
    TTFont("NotoSans", FONT_PATH)
)

style = ParagraphStyle(
    "Chinese",
    fontName="NotoSans",
    fontSize=12,
    leading=18
)

def generate_pdf(path):

    story = []

    story.append(Paragraph("人工智能行业投资研究报告", style))
    story.append(Spacer(1, 20))

    for i in range(5):

        story.append(Paragraph(f"章节 {i+1} 行业分析", style))
        story.append(Spacer(1,10))

        for _ in range(6):

            text = long_paragraph().replace("\n","<br/>")
            story.append(Paragraph(text, style))
            story.append(Spacer(1,8))

        table_data = [["公司","行业","市值","增长率"]]

        for _ in range(6):

            table_data.append([
                random.choice(clients),
                random.choice(industries),
                str(random.randint(100,800))+"亿",
                str(random.randint(5,30))+"%"
            ])

        story.append(Paragraph("主要投资结论", style))

        story.append(Paragraph(
            "• 人工智能行业未来五年预计保持15%增长<br/>"
            "• 星云科技集团具备较强技术壁垒<br/>"
            "• HC AI Innovation Fund 已进行战略投资",
            style))

        # 关键修改部分
        table = Table(
            table_data,
            colWidths=[6*cm,4*cm,3*cm,3*cm]
        )

        table.setStyle(TableStyle([
            ('FONTNAME', (0,0), (-1,-1), 'NotoSans'),
            ('FONTSIZE', (0,0), (-1,-1), 10),
            ('ALIGN', (2,1), (-1,-1), 'RIGHT'),
            ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
            ('BACKGROUND', (0,0), (-1,0), colors.lightgrey)
        ]))

        story.append(table)
        story.append(Spacer(1,20))

    pdf = SimpleDocTemplate(path)

    pdf.build(story)
# Excel
def generate_excel(path):

    wb=Workbook()

    sheets=[
    "股票持仓",
    "债券持仓",
    "基金持仓",
    "行业配置",
    "历史收益",
    "风险指标"
    ]

    for i,name in enumerate(sheets):

        if i==0:
            ws=wb.active
            ws.title=name
        else:
            ws=wb.create_sheet(name)

        ws.append(["公司","行业","市值","持仓比例","收益率"])

        for _ in range(300):

            ws.append([
                random.choice(clients),
                random.choice(industries),
                random.randint(1000000000,50000000000),
                round(random.uniform(0.1,10),2),
                round(random.uniform(-10,25),2)
            ])

    wb.save(path)


# JSON
def generate_json(path):

    data=[]

    for _ in range(2000):

        item={
        "date":fake.date(),
        "company":random.choice(clients),
        "sector":random.choice(industries),
        "price":round(random.uniform(20,200),2),
        "volume":random.randint(100000,5000000),
        "market_cap":random.randint(1000000000,50000000000)
        }

        data.append(item)

    with open(path,"w",encoding="utf8") as f:
        json.dump(data,f,ensure_ascii=False,indent=2)


# PPT
def generate_ppt(path):

    prs=Presentation()

    for i in range(25):

        slide_layout=prs.slide_layouts[1]
        slide=prs.slides.add_slide(slide_layout)

        title=slide.shapes.title
        content=slide.placeholders[1]

        title.text=f"投资策略分析 {i+1}"
        content.text=generate_long_text(3)

    prs.save(path)


# TXT
def generate_txt(path):

    with open(path,"w",encoding="utf8") as f:

        for _ in range(200):
            f.write(long_paragraph())
            f.write("\n\n")


def generate_files():

    for i in tqdm(range(60)):
        generate_md(f"{ROOT}/02_行业研究/行业研究报告_{i+1}.md")


    for i in tqdm(range(40)):
        generate_docx(f"{ROOT}/03_客户与项目/客户投资分析报告_{i+1}.docx")

    for i in tqdm(range(35)):
        generate_pdf(f"{ROOT}/04_基金产品/基金投资研究报告_{i+1}.pdf")

    for i in tqdm(range(20)):
        generate_excel(f"{ROOT}/07_投资数据/投资组合数据_{i+1}.xlsx")

    for i in tqdm(range(15)):
        generate_json(f"{ROOT}/08_市场数据/股票市场历史数据_{i+1}.json")

    for i in tqdm(range(15)):
        generate_ppt(f"{ROOT}/09_内部培训/投资策略培训_{i+1}.pptx")

    for i in tqdm(range(15)):
        generate_txt(f"{ROOT}/10_内部邮件/内部讨论邮件_{i+1}.txt")


if __name__=="__main__":

    generate_files()

    print("金融RAG知识库数据生成完成")