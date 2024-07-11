import jiwer

# Abrimos archivos para leer los resultados y la transcripcion de referencia
# Es importante abrirlos en encodeamiento uft-8
model = "trained"

transFile = open(model + "_normalized.txt", "r", encoding="utf-8")
refsFile = open("references_normalized.txt", "r", encoding="utf-8")

resWordMetricsFile = open(model + "_results_word_metrics.txt", "a", encoding="utf-8")
resCharMetricsFile = open(model + "_results_char_metrics.txt", "a", encoding="utf-8")

trans = transFile.readlines()
refs = refsFile.readlines()

result = jiwer.process_words(refs, trans)
resWordMetricsFile.write(jiwer.visualize_alignment(result))

result = jiwer.process_characters(refs, trans)
resCharMetricsFile.write(jiwer.visualize_alignment(result))

print("Metrics calculated")

refsFile.close()
transFile.close()
resWordMetricsFile.close()
resCharMetricsFile.close()