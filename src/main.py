import json
import os
import uvicorn
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from wtpsplit import SaT

# ============ 配置 ============
service_config_path = "./service_config.json"
service_config = json.load(open(service_config_path, "r", encoding="utf-8"))

model_folder = service_config.get("model_folder", "./models")
model_name = service_config.get("model_name", "sat-12l-sm")
model_path = f"{model_folder}/{model_name}"

# ============ 加载模型 ============
if not Path(model_path).exists():
    print(f"未找到模型 {model_path}, 尝试自动下载 segment-any-text/{model_name}")
    if os.system(f"hf download segment-any-text/{model_name} --local-dir {model_path}") != 0:
        print(f"未找到模型 {model_name} 在 {model_path}, 且自动下载 segment-any-text/{model_name} 失败")
        exit(1)


print(f"加载模型: {model_path}")
model = SaT(model_path)
model.half().to("cuda")
print("模型加载完成")

app = FastAPI(title="Sat Service")


# ============ 请求模型 ============
class SentenceSpiltRequest(BaseModel):
    text: str = Field(description="需要切分的文本")


# ============ 接口 ============

@app.get("/status")
async def health():
    """健康检查"""
    return {"status": "ok"}


@app.post("/sentence_spilt")
async def sentence_spilt(request: SentenceSpiltRequest):
    """
    句子分割接口
    输入文本，返回分割后的句子列表
    """
    try:
        sentences = list(model.split(request.text))
        return {"sentences": sentences}
    except Exception as e:
        # 发生错误时返回 500 状态码及错误信息
        raise HTTPException(status_code=500, detail=f"句子分割失败: {str(e)}")


# ============ 启动 ============
if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=service_config.get("port", 8003),
        log_level="info"
    )