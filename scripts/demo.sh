#!/usr/bin/env bash
# Démo live bout-en-bout : santé -> ingestion -> 3 scénarios de questions.
#
# Usage :
#   ./scripts/demo.sh              # scénarios complets
#   ./scripts/demo.sh --quick      # sans métriques (réponses rapides)
set -euo pipefail
cd "$(dirname "$0")/.."

BASE=${BASE:-http://localhost:8000}
METRIC_FLAG="true"
QUICK="false"
if [ "${1:-}" = "--quick" ]; then
  METRIC_FLAG="false"
  QUICK="true"
fi

separator() {
  echo
  echo "=================================================================="
}

(separator
echo "1. Santé de l'API"
curl -fsS "${BASE}/health" | python3 -m json.tool)

(separator
echo "2. Ingestion du corpus (documents/ -> index FAISS)"
curl -fsS -X POST "${BASE}/ingest" \
  -F "files=@documents/evenements-publics-openagenda-05ba8870dc7c40dfa04aca552f7a1bb8.md" \
  | python3 -m json.tool)

question() {
  local titre="$1"; local question="$2"
  separator
  echo "3. ${titre}"
  echo "   Question : ${question}"
  curl -fsS -X POST "${BASE}/prompt" \
    -H "Content-Type: application/json" \
    -d "$(python3 -c '
import json, sys
print(json.dumps({
    "question": sys.argv[1],
    "mode": "auto",
    "include_metrics": sys.argv[2] == "true",
    "include_ragas": False,
}))
' "${question}" "${METRIC_FLAG}")" \
    | python3 -c '
import json, sys
r = json.load(sys.stdin)
print(f"   Mode        : {r[\"mode\"]} (rag_used={r[\"rag_used\"]})")
print(f"   Réponse     : {r[\"answer\"]}")
print(f"   Temps (ms)  : {r[\"elapsed_ms\"]:.0f}")
for c in r["contexts"][:2]:
    d = c.get("document") or {}
    print(f"   Source      : {d.get(\"title\", \"?\")} (score {c[\"score\"]:.3f})")
print(f"   Métriques   : {json.dumps(r[\"metrics\"], ensure_ascii=False)}" if r.get("metrics") else "   Métriques   : désactivées")
'
}

question "Scénario A — événement culturel (mode RAG attendu)" \
  "Quels concerts de jazz sont organisés à Paris ?"
question "Scénario B — question métier sur le corpus" \
  "Quels spectacles gratuits ont lieu ce week-end et où ?"
question "Scénario C — question générale (mode DIRECT attendu)" \
  "Peux-tu me présenter La Réunion en deux phrases ?"
question "Scénario D — hors corpus (le RAG ne doit pas inventer)" \
  "Quel est le programme du festival de Cannes 2030 ?"

separator
echo "Démo terminée. Swagger UI : ${BASE}/docs"
