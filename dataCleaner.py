from pathlib import Path
import pyarrow.parquet as pq

projectDirectory = Path(__file__).parent

files = [
    "train-00000-of-00001.parquet",
    "valid-00000-of-00001.parquet",
    "test-00000-of-00001.parquet",
]

unwantedColumns = {
    "url",
    "date",
    "authors",
    "source",
    "content_original",
    "source_url",
    "ID",
    "bias_text",
    "topic"
}

outputDirectory = projectDirectory / "cleaned"
outputDirectory.mkdir(exist_ok = True)

for filename in files:
    inputPath = projectDirectory / filename
    outputPath = outputDirectory / filename

    allCols = pq.read_schema(inputPath).names
    keepCols = [col for col in allCols if col not in unwantedColumns]

    table = pq.read_table(inputPath, columns = keepCols)
    pq.write_table(table, outputPath)