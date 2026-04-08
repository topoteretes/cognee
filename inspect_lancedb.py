import asyncio
import lancedb

DB_PATH = "/Users/igorilic/Desktop/cognee/cognee/.cognee_system/databases/ef1fb438-8f24-453c-84f9-156779cdadfd/61b15d4f-d8da-5078-90d8-70817d811b37.lance.db"


async def main():
    db = await lancedb.connect_async(DB_PATH)
    table_names = await db.table_names()

    if not table_names:
        print("No tables found.")
        return

    for name in table_names:
        print(f"\n{'=' * 60}")
        print(f"TABLE: {name}")
        print("=" * 60)

        t = await db.open_table(name)
        schema = await t.schema()
        row_count = await t.count_rows()
        version = await t.version()

        print(f"Version : {version}")
        print(f"Rows    : {row_count}")
        print(f"Schema  :\n{schema}\n")

        if row_count == 0:
            print("(empty)")
            continue

        rows = (await t.to_arrow()).to_pylist()
        for i, row in enumerate(rows):
            print(f"--- Row {i + 1} ---")
            for key, value in row.items():
                if key == "vector":
                    print(f"  vector: [{value[0]:.4f}, {value[1]:.4f}, ... ] (len={len(value)})")
                else:
                    print(f"  {key}: {value}")


asyncio.run(main())
