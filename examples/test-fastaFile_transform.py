# %%python2
# from pymisca.atto_job import ModuleJob
from pymisca.atto_jobs_list.fastaFile_transform import fastaFile_transform
TEST_FASTA = "/work/reference-database/salmonella-typhimurium/G00000001/ANNOTATION_FASTA/CDNA.fasta"
OFNAME = "test-out.fasta"

fastaFile_transform({
    "INPUT_FILE":TEST_FASTA,
    "OFNAME": OFNAME,
    "FORCE":1,
    "FUNC_OBJ2LINES":lambda obj:obj[-1] if obj[0].lstrip(">").startswith("gene-SL") else [],
})