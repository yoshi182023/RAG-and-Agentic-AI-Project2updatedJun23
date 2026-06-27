# 从 IBM watsonx AI SDK 导入凭证类，用于连接云端大模型服务
from ibm_watsonx_ai import Credentials
# 导入模型推理类，用于调用并生成文本
from ibm_watsonx_ai.foundation_models import ModelInference
# 导入文本生成参数的元数据名称（如最大 token 数、解码方式等）
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
# 导入解码方法枚举（如贪心解码 GREEDY）
from ibm_watsonx_ai.foundation_models.utils.enums import DecodingMethods
# 导入 Pydantic：BaseModel 定义数据模型，Field 定义字段，ValidationError 捕获校验错误
from pydantic import BaseModel, Field, ValidationError
# 导入类型注解：List 表示列表，Optional 表示字段可为空
from typing import List, Optional
# 导入 json 模块，用于读写 JSON 文件
import json
# 导入 os 模块，用于检查文件是否存在等操作系统相关操作
import os
# 导入 shutil 模块，用于复制/备份文件
import shutil
# 导入 io 模块（预留，可用于字符串流等操作）
import io
# 导入 sys 模块，用于解析命令行参数
import sys
# 导入 unittest 单元测试框架
import unittest
# 导入 patch，用于在测试中模拟（mock）函数行为
from unittest.mock import patch

# 结构化餐厅数据的主 JSON 文件路径
FILEPATH = 'structured_restaurant_data.json'
# 写入新数据前，原文件的备份路径
BACKUP_PATH = 'structured_restaurant_data.json.bak'
# 示例餐厅段落文本，用于 one-shot prompt 中的示范输入
EXAMPLE_RESTAURANT_PARAGRAPH = (
    'Down in **Santa Monica**, **Mar de Cortez** serves as a **sun-drenched**, '
    '**casual taqueria** specializing in **Baja-style seafood**. With a **4.2/5** '
    'rating, it captures the salt-air energy of the coast through its signature '
    'beer-battered snapper tacos and zesty octopus ceviche, making it a premier '
    'spot for open-air dining near the pier. Price range: $.'
)
# 与示例段落对应的期望 JSON 输出，供 LLM 参考格式
EXAMPLE_OUTPUT = """
{
    "name": "Mar de Cortez",
    "location": "Santa Monica",
    "type": "casual taqueria",
    "food_style": "Baja-style seafood",
    "rating": 4.2,
    "price_range": 1,
    "signatures": [
        "beer-battered snapper tacos",
        "zesty octopus ceviche"
    ],
    "vibe": "salt-air energy",
    "environment": "a premier sun-drenched spot for open-air dining near the pier.",
    "shortcomings": []
}
"""


# 定义餐厅数据的 Pydantic 数据模型（JSON Schema），用于校验 LLM 输出是否合法
class Restaurant(BaseModel):
    name: str                          # 餐厅名称（必填）
    location: str                      # 所在位置（必填）
    type: str                          # 餐厅类型，如 bistro、taqueria（必填）
    food_style: str                    # 菜系/食物风格（必填）
    rating: Optional[float] = None     # 评分，可为空
    price_range: Optional[int] = None  # 价格档位（$ 的数量），可为空
    signatures: List[str] = Field(default_factory=list)   # 招牌菜列表，默认空列表
    vibe: Optional[str] = None         # 氛围/气质描述，可为空
    environment: str                   # 环境描述（必填）
    shortcomings: List[str] = Field(default_factory=list)  # 缺点列表，默认空列表


# ---------- 第 1 课练习 2：LLM 基础调用 ----------
def llm_model(system_msg, prompt_txt, params=None):
    """调用 IBM Granite 大模型，根据系统消息和用户提示生成文本回复。"""
    # 使用的模型 ID（Granite 系列，成本较低、适合结构化提取任务）
    model_id = "ibm/granite-4-h-small"
    # Skills Network 实验环境提供的项目 ID
    project_id = "skills-network"
    # 创建 watsonx 云端 API 凭证（实验环境无需额外 API Key）
    credentials = Credentials(url="https://us-south.ml.cloud.ibm.com")
    # 若未传入自定义参数，则使用默认：贪心解码 + 最多生成 1000 个新 token
    parameters = params or {
        GenParams.DECODING_METHOD: DecodingMethods.GREEDY,
        GenParams.MAX_NEW_TOKENS: 1000,
    }

    # 实例化模型推理对象
    model = ModelInference(
        model_id=model_id,
        credentials=credentials,
        project_id=project_id,
        params=parameters,
    )

    # 构造聊天消息列表：先 system 再 user（标准 chat 格式）
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": prompt_txt},
    ]
    # 调用模型 chat 接口获取回复
    response = model.chat(messages=messages)
    # 从回复中提取模型生成的文本内容并返回
    return response["choices"][0]["message"]["content"]


# ---------- 第 1 课练习 3：Prompt 工程 ----------
def restaurant_data_structure_prompt_generation(restaurant_paragraph):
    """根据餐厅段落生成发给 LLM 的系统消息和用户提示词（one-shot prompting）。"""
    # 系统消息：告诉 LLM 扮演数据提取助手，并规定输出格式和字段要求
    base_system_msg = """
    You are a data extraction assistant specialized in parsing restaurant descriptions.
    Your task is to extract structured information from unstructured restaurant text.
    Always respond with valid JSON only, without any additional explanation or markdown formatting.
    For the price_range field, convert dollar signs ($, $$, $$$) into an integer representing the number of dollar symbols.
    Include the following fields: name, location, type, food_style, rating, price_range, signatures (list), vibe, environment, shortcomings (list).
    """
    # 用户提示词：包含待提取的段落 + 示例输入/输出，引导 LLM 按同样格式返回 JSON
    base_user_prompt = f"""
    Task:
    Extract structured restaurant information from the following restaurant description and return it as valid JSON.
    Use the example below as a guide for the expected output format.

    Restaurant description:
    {restaurant_paragraph}

    Example:
    Input Restaurant Description: {EXAMPLE_RESTAURANT_PARAGRAPH}
    Output:
    {EXAMPLE_OUTPUT}
    """
    # 返回 (系统消息, 用户提示词) 元组，供 llm_model 使用
    return base_system_msg, base_user_prompt


# ---------- 第 1 课练习 4：JSON 自动修复 ----------
def JSON_auto_repair_prompts(response, error_message):
    """当 JSON 校验失败时，生成用于让 LLM 自动修复 JSON 的提示词。"""
    # 系统消息：定义 LLM 为 JSON 修复专家
    auto_repair_system_msg = """
    You are a JSON repair expert. Your sole task is to fix invalid JSON outputs so they conform to the required schema.
    Return only valid JSON without any additional text or markdown formatting.
    """
    # 用户提示词：附上错误的 JSON 和 Pydantic 校验错误信息，要求 LLM 修正
    auto_repair_prompt = f"""
    The following JSON output is invalid and needs to be corrected.

    Invalid JSON output:
    {response}

    Validation error message:
    {error_message}

    Please fix the JSON output so it conforms to the required schema and return only the corrected JSON.
    """
    # 返回修复用的 (系统消息, 用户提示词)
    return auto_repair_system_msg, auto_repair_prompt


# ---------- CLI 辅助函数：读写 JSON 数据库 ----------
def load_data(file_path):
    """从 JSON 文件加载餐厅列表；文件不存在时返回空列表。"""
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as file:
            return json.load(file)
    return []


def save_data(data, file_path, backup_path):
    """保存餐厅列表到 JSON 文件；若原文件存在则先备份。"""
    if os.path.exists(file_path):
        shutil.copy2(file_path, backup_path)
    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4)


def show_restaurant_card(record, index):
    """格式化打印单条餐厅记录的详细信息。"""
    print(f"\n--- Record #{index} ---")
    for key, value in record.items():
        print(f"{key}: {value}")


# ---------- 第 3 课：结构化提取 + 写入数据库 ----------
def _structure_and_validate_paragraph(paragraph):
    """内部辅助函数：将段落转为 JSON 并循环校验/修复，直到通过 Restaurant 模型校验。"""
    # 为当前段落生成提取用的 prompt
    base_system_msg, base_user_prompt = restaurant_data_structure_prompt_generation(paragraph)
    # 调用 LLM 得到初始 JSON 字符串
    response = llm_model(base_system_msg, base_user_prompt)

    # 循环：校验通过则跳出，失败则调用修复 prompt 让 LLM 重新生成
    while True:
        try:
            # 尝试用 Pydantic 模型校验 JSON 字符串
            Restaurant.model_validate_json(response)
            break  # 校验成功，退出循环
        except ValidationError as error:
            # 校验失败：生成修复 prompt，再次调用 LLM
            repair_system_msg, repair_prompt = JSON_auto_repair_prompts(response, error.json())
            response = llm_model(repair_system_msg, repair_prompt)

    # 将最终合法的 JSON 字符串解析为 Python 字典并返回
    return json.loads(response)


def new_data_entry_process(paragraph, itemId):
    """
    核心业务流程：把新的餐厅段落结构化，写入 JSON 数据库。
    参数 paragraph: 餐厅描述文本；itemId: 该条记录的唯一 ID。
    """
    # 第一步：用 LLM 提取并校验，得到餐厅字典（尚无 itemId）
    restaurant_dict = _structure_and_validate_paragraph(paragraph)
    # 第二步：写入指定的 itemId，与评论数据等保持一致
    restaurant_dict["itemId"] = itemId

    # 第三步：读取现有数据、追加新记录、备份并保存
    restaurant_data = load_data(FILEPATH)
    restaurant_data.append(restaurant_dict)
    save_data(restaurant_data, FILEPATH, BACKUP_PATH)

    # 返回刚添加的餐厅字典
    return restaurant_dict


# ---------- 命令行交互界面 ----------
def manage_restaurants(file_path, backup_path):
    """命令行交互式餐厅数据库管理界面。"""
    while True:
        # 从 JSON 文件加载当前所有餐厅记录
        data = load_data(file_path)
        # 显示当前记录数和菜单选项
        print(f"\n🏨 RESTAURANT DATABASE | Records: {len(data)}")
        print("1. Browse All (Names)")
        print("2. View Detailed Record")
        print("3. Add New Restaurant")
        print("4. Edit Restaurant Info")
        print("5. Delete Restaurant")
        print("6. Exit")

        # 读取用户选择的操作编号
        choice = input("\nAction: ")

        if choice == '1':
            # 选项 1：浏览所有餐厅名称
            print("\n--- Current Listings ---")
            for index, record in enumerate(data):
                print(f"[{index}] {record.get('name', 'N/A')}")

        elif choice == '2':
            # 选项 2：查看某条记录的详细信息
            try:
                index = int(input("Enter record index: "))
            except ValueError:
                print("Invalid index.")
                continue
            if 0 <= index < len(data):
                show_restaurant_card(data[index], index)
            else:
                print("Invalid index.")

        elif choice in ['3', '4', '5']:
            # 写操作前要求用户确认，防止误改数据库
            print("\n❗ SECURITY WARNING: You are entering write-mode.")
            print("Changes will be saved to the database immediately.")
            confirm = input("Are you sure? (type 'yes' to proceed): ").lower()
            if confirm != 'yes':
                print("Operation cancelled.")
                continue

            if choice == '3':
                # 选项 3：添加新餐厅（new_data_entry_process 内部已完成保存）
                itemId = 1000000 + len(data) + 1
                paragraph = input("Enter restaurant description: ").strip()
                if not paragraph:
                    print("Description cannot be empty.")
                    continue
                new_data_entry_process(paragraph, itemId)
                print("✅ Restaurant added.")

            elif choice == '4':
                # 选项 4：编辑餐厅信息
                try:
                    index = int(input("Enter record index to edit: "))
                except ValueError:
                    print("Invalid index.")
                    continue
                if not (0 <= index < len(data)):
                    print("Invalid index.")
                    continue
                record = data[index]
                for key in list(record.keys()):
                    if key == "itemId":
                        continue
                    new_value = input(f"{key} [{record[key]}]: ").strip()
                    if new_value:
                        record[key] = new_value
                save_data(data, file_path, backup_path)
                print("✅ Record updated.")

            elif choice == '5':
                # 选项 5：删除餐厅记录
                try:
                    index = int(input("Enter record index to delete: "))
                except ValueError:
                    print("Invalid index.")
                    continue
                if 0 <= index < len(data):
                    data.pop(index)
                    save_data(data, file_path, backup_path)
                    print("✅ Record deleted.")
                else:
                    print("Invalid index.")

        elif choice == '6':
            # 选项 6：退出程序
            break
        else:
            print("Invalid input.")


# ---------- 单元测试 ----------
class TestRestaurantDataManagement(unittest.TestCase):
    """单元测试类：用 mock 模拟 LLM，无需真实调用 API 即可验证逻辑。"""

    # 模拟 LLM 返回的合法 JSON 样本
    SAMPLE_RESPONSE = """
    {
        "name": "Mar de Cortez",
        "location": "Santa Monica",
        "type": "casual taqueria",
        "food_style": "Baja-style seafood",
        "rating": 4.2,
        "price_range": 1,
        "signatures": [
            "beer-battered snapper tacos",
            "zesty octopus ceviche"
        ],
        "vibe": "salt-air energy",
        "environment": "a premier sun-drenched spot for open-air dining near the pier.",
        "shortcomings": []
    }
    """

    def setUp(self):
        """每个测试方法运行前：创建临时测试用的 JSON 文件。"""
        self.test_filepath = "test_structured_restaurant_data.json"
        self.test_backup_path = "test_structured_restaurant_data.json.bak"
        # 预设一条已有餐厅数据
        self.existing_data = [
            {
                "name": "The Gilded Artichoke",
                "location": "Silver Lake",
                "type": "upscale bistro",
                "food_style": "Farm-to-Table Californian",
                "rating": 4.5,
                "price_range": 4,
                "signatures": ["lavender-rubbed roasted chicken"],
                "vibe": "bohemian chic",
                "environment": "a high-end greenhouse",
                "shortcomings": [],
                "itemId": 1000001,
            }
        ]
        # 将预设数据写入临时文件
        with open(self.test_filepath, "w", encoding="utf-8") as file:
            json.dump(self.existing_data, file, indent=4)

    def tearDown(self):
        """每个测试方法运行后：删除临时测试文件。"""
        for path in (self.test_filepath, self.test_backup_path):
            if os.path.exists(path):
                os.remove(path)

    def test_prompt_generation_returns_messages(self):
        """测试：prompt 生成函数应包含关键角色说明和示例餐厅名。"""
        system_msg, user_prompt = restaurant_data_structure_prompt_generation(
            EXAMPLE_RESTAURANT_PARAGRAPH
        )
        self.assertIn("data extraction assistant", system_msg.lower())
        self.assertIn(EXAMPLE_RESTAURANT_PARAGRAPH, user_prompt)
        self.assertIn("Mar de Cortez", user_prompt)

    def test_json_auto_repair_prompts(self):
        """测试：JSON 修复 prompt 应包含错误 JSON 和错误信息。"""
        system_msg, repair_prompt = JSON_auto_repair_prompts('{"name": "Test"}', "missing field")
        self.assertIn("JSON repair expert", system_msg)
        self.assertIn('{"name": "Test"}', repair_prompt)
        self.assertIn("missing field", repair_prompt)

    @patch(f"{__name__}.llm_model")
    def test_new_data_entry_process(self, mock_llm):
        """测试：新增数据流程应正确写入 itemId、备份文件并追加记录。"""
        # 让 mock 的 llm_model 直接返回预设 JSON，不调用真实 API
        mock_llm.return_value = self.SAMPLE_RESPONSE

        # 临时把 FILEPATH/BACKUP_PATH 指向测试文件
        # 使用 __name__ 确保无论以脚本还是模块方式运行，patch 都能命中正确目标
        with patch(f"{__name__}.FILEPATH", self.test_filepath), patch(
            f"{__name__}.BACKUP_PATH", self.test_backup_path
        ):
            result = new_data_entry_process(EXAMPLE_RESTAURANT_PARAGRAPH, 1000999)

        # 断言返回结果字段正确
        self.assertEqual(result["name"], "Mar de Cortez")
        self.assertEqual(result["itemId"], 1000999)
        # 断言备份文件已创建
        self.assertTrue(os.path.exists(self.test_backup_path))

        # 读取保存后的文件，断言共 2 条记录且最后一条 itemId 正确
        with open(self.test_filepath, "r", encoding="utf-8") as file:
            saved_data = json.load(file)

        self.assertEqual(len(saved_data), 2)
        self.assertEqual(saved_data[-1]["itemId"], 1000999)
        # 断言 LLM 只被调用了一次（校验一次通过，无需修复）
        mock_llm.assert_called_once()


# ---------- 程序入口（放在文件最末尾）----------
if __name__ == "__main__":
    # 默认运行单元测试；加 --cli 参数启动交互式管理界面
    if len(sys.argv) > 1 and sys.argv[1] == "--cli":
        manage_restaurants(FILEPATH, BACKUP_PATH)
    else:
        unittest.main()
