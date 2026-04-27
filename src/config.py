from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    anthropic_api_key: str
    order_query_service_url: str = "http://localhost:8091"
    redis_url: str = "redis://localhost:6379/0"
    database_url: str = "postgresql://postgres:postgres@localhost:5432/customer_agent"
    langchain_tracing_v2: bool = True
    langchain_api_key: str = ""
    langchain_project: str = "cloudorder-customer-agent"

    model_config = SettingsConfigDict(env_file=".env")


settings = Settings()
