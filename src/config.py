from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    anthropic_api_key: str
    order_query_service_url: str
    redis_url: str
    database_url: str
    langchain_tracing_v2: bool = True
    langchain_api_key: str = ""
    langchain_project: str = "cloudorder-customer-agent"

    model_config = SettingsConfigDict(env_file=".env")


settings = Settings()
