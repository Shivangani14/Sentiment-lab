#!/usr/bin/env bash
# Full regression pass over the Sentiment Lab HTTP surface.
set -u
B=http://localhost:5000
pass=0; fail=0
absent() { # absent <name> <pattern> <actual>   -- passes when NOT found
  if printf '%s' "$3" | grep -q "$2"; then
    printf '  FAIL  %s\n        must NOT contain: %s\n' "$1" "$2"; fail=$((fail+1))
  else
    printf '  PASS  %s\n' "$1"; pass=$((pass+1))
  fi
}
chk() { # chk <name> <expected-substring> <actual>
  if printf '%s' "$3" | grep -q "$2"; then
    printf '  PASS  %s\n' "$1"; pass=$((pass+1))
  else
    printf '  FAIL  %s\n        expected to contain: %s\n        got: %s\n' "$1" "$2" "$(printf '%s' "$3" | head -c 300)"; fail=$((fail+1))
  fi
}
code() { curl -s -o /dev/null -w '%{http_code}' "$@"; }

echo "== pages & redirects =="
chk "GET / redirects"        "302" "$(code $B/)"
chk "GET /bulk redirects"    "302" "$(code $B/bulk)"
chk "GET /analyze is 200"    "200" "$(code $B/analyze)"
chk "404 page renders"       "404" "$(code $B/no-such-page)"
chk "/analyze has 4 tabs"    "tab-blog" "$(curl -s $B/analyze)"
absent "no API key in HTML"      "AIza\|YOUTUBE_API_KEY=" "$(curl -s $B/analyze)"
absent "no API key in JS"        "AIza" "$(curl -s $B/static/js/analyze.js)"

echo "== status =="
S=$(curl -s $B/api/status)
chk "status ok"              '"success": true' "$S"
chk "vader live"             '"vader": true' "$S"
absent "no key leaked"           "AIza" "$S"

echo "== text =="
chk "positive english"  '"sentiment": "POSITIVE"' "$(curl -s -X POST $B/api/analyze/text -H 'Content-Type: application/json' -d '{"text":"I absolutely love this product!"}')"
chk "negative english"  '"sentiment": "NEGATIVE"' "$(curl -s -X POST $B/api/analyze/text -H 'Content-Type: application/json' -d '{"text":"This is absolutely terrible and awful."}')"
chk "hindi detected"    '"language": "Hindi"'     "$(curl -s -X POST $B/api/analyze/text -H 'Content-Type: application/json' -d '{"text":"बहुत अच्छा वीडियो है।"}')"
chk "hinglish detected" '"language": "Hinglish"'  "$(curl -s -X POST $B/api/analyze/text -H 'Content-Type: application/json' -d '{"text":"Bhai kya hi video hai"}')"
chk "empty rejected"    '"success": false'        "$(curl -s -X POST $B/api/analyze/text -H 'Content-Type: application/json' -d '{"text":"  "}')"
absent "no traceback on bad body" "Traceback" "$(curl -s -X POST $B/api/analyze/text -H 'Content-Type: application/json' -d '{}')"
chk "bad json handled"  '"success": false' "$(curl -s -X POST $B/api/analyze/text -H 'Content-Type: application/json' -d 'not json')"

echo "== legacy v1 endpoints =="
chk "POST /predict"     '"compound"' "$(curl -s -X POST $B/predict -H 'Content-Type: application/json' -d '{"text":"I love this"}')"
L=$(curl -s -X POST $B/bulk-upload -F "file=@tests/test_sentiments.csv")
chk "POST /bulk-upload" '"success": true' "$L"
chk "v1 columns kept"   'sentiment_label' "$L"
FID=$(printf '%s' "$L" | python3 -c "import json,sys;print(json.load(sys.stdin).get('file_id',''))" 2>/dev/null)
chk "v1 download works" "sentiment_compound" "$(curl -s $B/download/$FID | head -1)"
chk "bad download 404"  "404" "$(code $B/download/deadbeef)"

echo "== dataset =="
I=$(curl -s -X POST $B/api/dataset/inspect -F "file=@tests/test_mixed.csv")
chk "column auto-picked" '"suggested_column": "review_text"' "$I"
UPL=$(printf '%s' "$I" | python3 -c "import json,sys;print(json.load(sys.stdin)['upload_id'])")
J=$(curl -s -X POST $B/api/dataset/analyze -H 'Content-Type: application/json' -d "{\"upload_id\":\"$UPL\",\"text_column\":\"review_text\",\"limit\":\"all\"}" | python3 -c "import json,sys;print(json.load(sys.stdin)['job_id'])")
for i in $(seq 1 30); do sleep 1; R=$(curl -s $B/api/job/$J); printf '%s' "$R" | grep -q '"status": "running"' || break; done
chk "dataset job done"    '"status": "done"' "$R"
chk "original cols kept"  '"product"' "$R"
chk "blank row unscored"  'UNSCORED' "$R"
chk "row count preserved" '"total": 7' "$R"
chk "non-csv rejected"    '"success": false' "$(curl -s -X POST $B/api/dataset/inspect -F 'file=@README.md')"
chk "no file rejected"    '"success": false' "$(curl -s -X POST $B/api/dataset/inspect)"
chk "stale job 404"       '"success": false' "$(curl -s $B/api/job/nope)"

echo "== youtube (no key configured) =="
chk "youtube blocked cleanly" '"success": false' "$(curl -s -X POST $B/api/youtube/analyze -H 'Content-Type: application/json' -d '{"url":"https://www.youtube.com/watch?v=dQw4w9WgXcQ"}')"
absent "youtube no traceback"    "Traceback" "$(curl -s -X POST $B/api/youtube/analyze -H 'Content-Type: application/json' -d '{"url":"garbage"}')"

echo "== blog =="
chk "loopback refused"    "Private and internal" "$(curl -s -X POST $B/api/blog/analyze -H 'Content-Type: application/json' -d '{"url":"http://127.0.0.1:5000/analyze"}')"
chk "file:// refused"     "Only http" "$(curl -s -X POST $B/api/blog/analyze -H 'Content-Type: application/json' -d '{"url":"file:///etc/passwd"}')"
chk "javascript: refused" "Only http" "$(curl -s -X POST $B/api/blog/analyze -H 'Content-Type: application/json' -d '{"url":"javascript:alert(1)"}')"
chk "cloud metadata refused" "Private and internal" "$(curl -s -X POST $B/api/blog/analyze -H 'Content-Type: application/json' -d '{"url":"http://169.254.169.254/latest/meta-data/"}')"
chk "private range refused"  "Private and internal" "$(curl -s -X POST $B/api/blog/analyze -H 'Content-Type: application/json' -d '{"url":"http://10.0.0.5/admin"}')"
chk "bad host refused"       "resolve" "$(curl -s -X POST $B/api/blog/analyze -H 'Content-Type: application/json' -d '{"url":"https://nope-xyz-123-abc.invalid/p"}')"

echo
echo "==== $pass passed, $fail failed ===="
[ "$fail" -eq 0 ]
