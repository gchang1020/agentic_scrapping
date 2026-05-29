import os
import time

# pip install anthropic google_genai
import anthropic
from google import genai as google_genai
from google.genai import types

env_name_anthropic = "ANTHROPIC_API_KEY"
env_name_gemini = "GEMINI_API_KEY"

if not os.getenv(env_name_anthropic):
	print(f"no {env_name_anthropic} defined yet")
	raise SystemExit(f"Run 'setx {env_name_anthropic} \"xxxxx\"'")
if not os.getenv("GEMINI_API_KEY"):
	print(f"no {env_name_gemini} defined yet")
	raise SystemExit(f"Run 'setx {env_name_gemini} \"xxxxx\"'")

CONFIG = {
    # Claude Pro (CLI) vs. Anthropic API
    # - Pro: chat w/ claude.ai, flat rate, no below "client" in code (although Claude Code can generate code)
    # - CLI: same as Pro, like r = subprocess.run(["claude", "-p", myprompt], capture_output=True, text=True, ...)
    # - API: via console.anthropic.com, per-per-token, client = anthropic.Anthropic(api_key="claude_pro_login")

    "claude": {
        "api_key": os.getenv(env_name_anthropic),
        # alternative: "claude-3-5-sonnet-20241022"
        "model": "claude-haiku-4-5-20251001",
        "cooldown": 3,
        # client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        "client_factory": lambda key: anthropic.Anthropic(api_key=key)
    },

    "gemini": {
        "api_key": os.getenv(env_name_gemini),
        # as said allows 20 requests / min, and 250-1000 requests / day
        "model": "gemini-2.5-flash",
        # 10 requests per minute (very safe for free tier)
        "cooldown": 10,
        "client_factory": lambda key: google_genai.Client(api_key=key)
    }
}
    
"""
 usage of garys_llm:
 from garys_llm import myLLM
 llm = myLLM("claude")
 print(llm.model)
"""
class myLLM:
    def __init__(self, provider):
        if provider not in CONFIG:
            raise ValueError(f"Unknown provider: {provider}")

        self.provider = provider
        self.cfg = CONFIG[provider]

        self.client = self.cfg["client_factory"](self.cfg["api_key"])
        self.model = self.cfg["model"]
        self.cooldown = self.cfg["cooldown"]

    def wait(self):
        time.sleep(self.cooldown)

    def generate(self, prompt, temperature = 0, max_tokens = 2048):
        if self.provider == "claude":
            response = self.client.messages.create(
                model = self.model,
                max_tokens = max_tokens,
                temperature = temperature, # controls randomness
                messages = [{"role": "user", "content": prompt}]
            )

            text = "".join(block.text for block in response.content if block.type == "text")

        elif self.provider == "gemini":
            response = self.client.models.generate_content(
                model = self.model,
                contents = prompt,
                config = types.GenerateContentConfig(
                    temperature = temperature, # controls randomness
                    max_output_tokens = max_tokens
                )
            )
            
            # response_text = response.text
            if hasattr(response, "text") and response.text:
                text = response.text
            else:
                text = response.candidates[0].content.parts[0].text

        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

        time.sleep(self.cooldown)
        return text

    """
    return a CrewAI LLM object built from this config
    requires: pip install crewai
    usage: bot = myLLM(provider); agent = Agent(role="...", llm=bot.to_crewai())
        - agent here is making decisions autonomously
        - the framework needs to own the LLM
    """
    def to_crewai(self):
        from crewai import LLM as CrewAILLM
        return CrewAILLM(
            model = self.cfg["model"],
            api_key = self.cfg["api_key"],
            temperature = 0,
            max_tokens = 2048,
        )

    """
    return a LangChain-compatible Runnable wrapping self.generate()
    requires: pip install langchain-core
    usage: bot = myLLM(provider); chain = prompt | bot.to_langchain() | StrOutputParser()
        - here, you are the orchestrator, "|" pipes the result in a chain
        - each step is a plain function you control/write
        - inside each function you just call bot.generate() yourself
    """
    def to_langchain(self):
        from langchain_core.runnables import RunnableLambda
        return RunnableLambda(
            lambda inp: self.generate(
                # inp is whatever the previous step in the chain passes in
                inp if isinstance(inp, str) else inp.text
            )
        )