import asyncio

from llm_bridge import get_file_type


async def main():
    file_type, sub_type = await get_file_type(
        "file_url" # replace this
    )
    print(file_type, sub_type)

if __name__ == "__main__":
    asyncio.run(main())
