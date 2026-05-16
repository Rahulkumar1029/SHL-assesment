from src.config.llm import get_llm,get_structured_llm,llm1,llm2


llm_res=llm1.invoke("What is linear")
print("llm1 result",llm_res.content)

llm2_res=llm2.invoke("What is ml")
print("llm2 result",llm2_res.content)