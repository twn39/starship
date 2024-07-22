from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    deep_seek_api_key: str
    open_router_api_key: str
    tongyi_api_key: str
    debug: bool

    model_config = SettingsConfigDict(env_file=('.env', '.env.local'), env_file_encoding='utf-8')


settings = Settings()
