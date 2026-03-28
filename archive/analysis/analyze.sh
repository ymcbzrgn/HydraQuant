#!/bin/bash
########################################################################
# FREQTRADE PHASE 20 — FULL LOG ANALYSIS v2
# Direkt dosyaya yazar, terminal output YOK
# Kullanım: nohup bash analyze_v2.sh &
# Çıktı: ~/freqtrade/FULL_ANALYSIS.txt
########################################################################

set -uo pipefail

OUT=~/freqtrade/FULL_ANALYSIS.txt
BOT_LOG=~/freqtrade/user_data/logs/freqtrade.log
DB=~/freqtrade/user_data/db/ai_data.sqlite
TMPDIR=$(mktemp -d)
trap "rm -rf $TMPDIR" EXIT

# HER ŞEY DOSYAYA — terminal'e hiçbir şey gitmesin
exec 1>"$OUT" 2>&1

cat <<'BANNER'
╔══════════════════════════════════════════════════════════════════╗
║  FREQTRADE PHASE 20 — FULL LOG ANALYSIS v2                     ║
╚══════════════════════════════════════════════════════════════════╝
BANNER
echo "Tarih: $(date '+%Y-%m-%d %H:%M:%S %Z')"
echo "Host:  $(hostname)"
echo ""

########################################################################
# 1. SERVİS DURUMLARI
########################################################################
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  1. SERVİS DURUMLARI"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
for svc in freqtrade freqtrade-rag freqtrade-scheduler freqtrade-models; do
  STATUS=$(systemctl is-active "$svc" 2>/dev/null || echo "NOT FOUND")
  SINCE=""
  if [ "$STATUS" = "active" ]; then
    SINCE=$(systemctl show "$svc" --property=ActiveEnterTimestamp --value 2>/dev/null || true)
  fi
  RESTARTS=$(systemctl show "$svc" --property=NRestarts --value 2>/dev/null || echo "?")
  MEM=$(systemctl show "$svc" --property=MemoryCurrent --value 2>/dev/null || echo "?")
  if [ "$MEM" != "?" ] && [ "$MEM" != "[not set]" ] && [ "$MEM" != "" ]; then
    if echo "$MEM" | grep -qP '^\d+$'; then
      MEM_MB=$((MEM / 1024 / 1024))
      MEM="${MEM_MB}MB"
    fi
  fi
  printf "  %-28s %-12s Restarts: %-4s Mem: %-10s Since: %s\n" "$svc" "[$STATUS]" "$RESTARTS" "$MEM" "$SINCE"
done
echo ""
echo "  System Uptime: $(uptime -p)"
echo "  Load Average:  $(cat /proc/loadavg)"
echo "  Disk Usage:    $(df -h / | tail -1 | awk '{print $5 " used (" $3 "/" $2 ")"}')"
echo "  RAM:           $(free -h | awk '/Mem:/{print $3 " / " $2}')"
echo ""

########################################################################
# 2. PORT KONTROL
########################################################################
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  2. PORT & ENDPOINT KONTROL"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
for port in 8080 8890 8891 8895; do
  LISTEN=$(ss -tlnp 2>/dev/null | grep ":${port} " | head -1 || echo "NOT LISTENING")
  printf "  Port %-6s → %s\n" "$port" "$LISTEN"
done
echo ""
echo "  API Health Checks:"
for endpoint in \
  "http://localhost:8891/health|RAG Health" \
  "http://localhost:8891/signal-health|Signal Health" \
  "http://localhost:8890/api/ai/agents|Agent Stats" \
  "http://localhost:8890/api/ai/cross-pair|Cross-Pair Intel"; do
  URL="${endpoint%%|*}"
  NAME="${endpoint##*|}"
  RESP=$(curl -s -o /dev/null -w "%{http_code}" --connect-timeout 5 "$URL" 2>/dev/null || echo "FAIL")
  printf "  %-25s → HTTP %s\n" "$NAME" "$RESP"
done
echo ""

########################################################################
# 3. LOG TOPLAMA — 50K SATIR
########################################################################
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  3. LOG TOPLAMA (son 50.000 satır / servis)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
if [ -f "$BOT_LOG" ]; then
  tail -n 50000 "$BOT_LOG" > "$TMPDIR/bot.log"
  echo "  freqtrade bot log:       $(wc -l < "$TMPDIR/bot.log") satır"
else
  touch "$TMPDIR/bot.log"
  echo "  freqtrade bot log:       DOSYA YOK!"
fi

for svc in freqtrade-rag freqtrade-scheduler freqtrade-models; do
  FNAME=$(echo "$svc" | tr '-' '_')
  journalctl -u "$svc" --no-pager -n 50000 --output=short-iso 2>/dev/null > "$TMPDIR/${FNAME}.log" || touch "$TMPDIR/${FNAME}.log"
  printf "  %-26s %s satır\n" "$svc:" "$(wc -l < "$TMPDIR/${FNAME}.log")"
done

if [ -s "$TMPDIR/bot.log" ]; then
  FIRST_DATE=$(head -1 "$TMPDIR/bot.log" | grep -oP '\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}' | head -1 || echo "?")
  LAST_DATE=$(tail -1 "$TMPDIR/bot.log" | grep -oP '\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}' | head -1 || echo "?")
  echo "  Bot log aralığı: $FIRST_DATE → $LAST_DATE"
fi
echo ""

########################################################################
# 4. HATA ANALİZİ
########################################################################
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  4. HATA & EXCEPTION ANALİZİ"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
for LOGFILE in "$TMPDIR"/bot.log "$TMPDIR"/freqtrade_rag.log "$TMPDIR"/freqtrade_scheduler.log "$TMPDIR"/freqtrade_models.log; do
  BASENAME=$(basename "$LOGFILE" .log)
  [ ! -s "$LOGFILE" ] && continue

  ERR_COUNT=$(grep -ciE 'error|exception|traceback|critical|fatal' "$LOGFILE" || echo 0)
  WARN_COUNT=$(grep -ciE 'warning|warn' "$LOGFILE" || echo 0)

  echo ""
  echo "  ── $BASENAME ──"
  echo "     Toplam ERROR/EXCEPTION/CRITICAL: $ERR_COUNT"
  echo "     Toplam WARNING:                  $WARN_COUNT"

  if [ "$ERR_COUNT" -gt 0 ] 2>/dev/null; then
    echo ""
    echo "     En sık hatalar (top 20):"
    grep -iE 'error|exception|critical|fatal' "$LOGFILE" \
      | sed 's/^[0-9T:.Z+\-]*//; s/^.*\] //' \
      | sed 's/[0-9a-f]\{8,\}/HASH/g; s/[0-9]\{1,\}\.[0-9]\{1,\}/N.N/g; s/pair=[^ ]*/pair=X/g' \
      | sort | uniq -c | sort -rn | head -20 \
      | while IFS= read -r line; do
          echo "     $line"
        done
  fi

  # Tüm unique traceback'ler
  TB_COUNT=$(grep -c "Traceback" "$LOGFILE" || echo 0)
  if [ "$TB_COUNT" -gt 0 ]; then
    echo ""
    echo "     Unique Traceback sonu satırları ($TB_COUNT toplam):"
    grep -B0 -A5 "Traceback" "$LOGFILE" \
      | grep -vE "^--|^$|Traceback" \
      | sed 's/^[0-9T:.Z+\- ]*//' \
      | sort -u | head -30 \
      | while IFS= read -r line; do
          echo "       $(echo "$line" | cut -c1-150)"
        done
  fi
done
echo ""

########################################################################
# 5. SİNYAL ANALİZİ
########################################################################
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  5. SİNYAL ANALİZİ"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "  5a. Sinyal Dağılımı (Bot log):"
for SIG in BULLISH BEARISH NEUTRAL; do
  COUNT=$(grep -c "$SIG" "$TMPDIR/bot.log" || echo 0)
  printf "     %-10s → %s adet\n" "$SIG" "$COUNT"
done

echo ""
echo "  5b. Signal Source Dağılımı:"
grep -oP '(EvidenceFirst|EvidenceEngine|MADAM|AgentPool|Voting|LegacyFallback|evidence_first|madam|legacy)' "$TMPDIR/bot.log" "$TMPDIR/freqtrade_rag.log" 2>/dev/null \
  | sed 's/.*://' | sort | uniq -c | sort -rn | head -10 \
  | while IFS= read -r line; do echo "     $line"; done

echo ""
echo "  5c. Confidence Dağılımı (GENİŞ REGEX):"
# Birden fazla pattern dene
cat "$TMPDIR/bot.log" "$TMPDIR/freqtrade_rag.log" 2>/dev/null \
  | grep -oP '(?:confidence|conf|"confidence")["\s:=]*\s*0\.\d+' \
  | grep -oP '0\.\d+' \
  | awk '{
      sum+=$1; count++;
      if($1<0.05) b0++;
      else if($1<0.10) b1++;
      else if($1<0.15) b2++;
      else if($1<0.20) b3++;
      else if($1<0.25) b4++;
      else if($1<0.30) b5++;
      else if($1<0.40) b6++;
      else if($1<0.50) b7++;
      else b8++;
    } END {
      if(count>0) {
        printf "     Toplam: %d sinyal\n", count;
        printf "     Ortalama: %.4f\n", sum/count;
        printf "     Min: -  Max: -  (awk ile hesaplanmadı)\n";
        printf "     [0.00-0.05): %d\n", b0;
        printf "     [0.05-0.10): %d\n", b1;
        printf "     [0.10-0.15): %d\n", b2;
        printf "     [0.15-0.20): %d\n", b3;
        printf "     [0.20-0.25): %d\n", b4;
        printf "     [0.25-0.30): %d\n", b5;
        printf "     [0.30-0.40): %d\n", b6;
        printf "     [0.40-0.50): %d\n", b7;
        printf "     [0.50+     ): %d\n", b8;
      } else {
        print "     Confidence verisi bulunamadı (regex eşleşmedi)";
        print "     DEBUG: İlk 5 confidence içeren satır:";
      }
    }'

# Regex eşleşmezse debug yardımı
CONF_MATCH=$(cat "$TMPDIR/bot.log" "$TMPDIR/freqtrade_rag.log" 2>/dev/null | grep -ciP 'confiden' || echo 0)
echo "     (Toplam 'confiden' içeren satır: $CONF_MATCH)"
echo ""
echo "     DEBUG — Confidence içeren ilk 10 satır (format tespiti):"
cat "$TMPDIR/bot.log" "$TMPDIR/freqtrade_rag.log" 2>/dev/null \
  | grep -iP 'confiden' | head -10 \
  | while IFS= read -r line; do
      echo "       $(echo "$line" | cut -c1-180)"
    done

echo ""
echo "  5d. Pair Bazında Sinyal Sayısı (top 25):"
grep -oP '[A-Z0-9]{2,}/USDT(?::USDT)?' "$TMPDIR/bot.log" 2>/dev/null \
  | sort | uniq -c | sort -rn | head -25 \
  | while IFS= read -r line; do echo "     $line"; done

echo ""
echo "  5e. Günlük Log Hacmi:"
grep -oP '^\d{4}-\d{2}-\d{2}' "$TMPDIR/bot.log" 2>/dev/null \
  | sort | uniq -c \
  | while IFS= read -r line; do echo "     $line"; done

echo ""
echo "  5f. Saatlik Sinyal Yoğunluğu (son 3 gün):"
grep -oP '^\d{4}-\d{2}-\d{2} \d{2}' "$TMPDIR/bot.log" 2>/dev/null \
  | sort | uniq -c | tail -72 \
  | while IFS= read -r line; do echo "     $line"; done
echo ""

########################################################################
# 6. EVIDENCE ENGINE ANALİZİ
########################################################################
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  6. EVIDENCE ENGINE ANALİZİ"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "  6a. Evidence Source Kullanımı:"
grep -oP 'sources?[=: ]+\d+/\d+' "$TMPDIR/freqtrade_rag.log" 2>/dev/null \
  | sort | uniq -c | sort -rn | head -10 \
  | while IFS= read -r line; do echo "     $line"; done

echo ""
echo "  6b. Evidence Sub-Question (son 30):"
grep -iE 'SubQ|sub.question|evidence.*score|evidence.*q[0-9]' "$TMPDIR/freqtrade_rag.log" 2>/dev/null \
  | tail -30 | while IFS= read -r line; do
      echo "     $(echo "$line" | cut -c1-160)"
    done

echo ""
echo "  6c. Evidence Audit Log (DB son 50):"
if [ -f "$DB" ]; then
  sqlite3 -header -column "$DB" "
    SELECT datetime(timestamp,'localtime') as ts,
           pair, signal, ROUND(confidence,4) as conf,
           regime, ROUND(max_confidence_cap,2) as cap
    FROM evidence_audit_log
    ORDER BY timestamp DESC LIMIT 50;
  " 2>/dev/null || echo "     QUERY FAILED"

  echo ""
  echo "  6d. Evidence — Sinyal Dağılımı (tüm kayıtlar):"
  sqlite3 -header -column "$DB" "
    SELECT signal,
           COUNT(*) as cnt,
           ROUND(AVG(confidence),4) as avg_conf,
           ROUND(MIN(confidence),4) as min_conf,
           ROUND(MAX(confidence),4) as max_conf
    FROM evidence_audit_log
    GROUP BY signal ORDER BY cnt DESC;
  " 2>/dev/null || echo "     QUERY FAILED"

  echo ""
  echo "  6e. Evidence — Günlük İstatistikler:"
  sqlite3 -header -column "$DB" "
    SELECT date(timestamp) as day,
           COUNT(*) as total,
           SUM(CASE WHEN signal='BULLISH' THEN 1 ELSE 0 END) as bull,
           SUM(CASE WHEN signal='BEARISH' THEN 1 ELSE 0 END) as bear,
           SUM(CASE WHEN signal='NEUTRAL' THEN 1 ELSE 0 END) as neut,
           ROUND(AVG(confidence),4) as avg_conf,
           ROUND(AVG(max_confidence_cap),2) as avg_cap
    FROM evidence_audit_log
    GROUP BY day ORDER BY day;
  " 2>/dev/null || echo "     QUERY FAILED"

  echo ""
  echo "  6f. Evidence — Pair bazında ortalama confidence (top 30):"
  sqlite3 -header -column "$DB" "
    SELECT pair,
           COUNT(*) as cnt,
           ROUND(AVG(confidence),4) as avg_conf,
           ROUND(MAX(confidence),4) as max_conf
    FROM evidence_audit_log
    GROUP BY pair ORDER BY avg_conf DESC LIMIT 30;
  " 2>/dev/null || echo "     QUERY FAILED"

  echo ""
  echo "  6g. Evidence — Sub-Score Örnekleri (son 10, JSON parse):"
  sqlite3 "$DB" "
    SELECT pair, signal, ROUND(confidence,3) as conf, sub_scores_json
    FROM evidence_audit_log ORDER BY timestamp DESC LIMIT 10;
  " 2>/dev/null | while IFS= read -r line; do echo "     $line"; done
fi
echo ""

########################################################################
# 7. LLM PROVIDER ANALİZİ
########################################################################
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  7. LLM PROVIDER ANALİZİ"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "  7a. HTTP Durum Kodları:"
for CODE in 200 400 401 403 413 429 500 502 503 504; do
  COUNT=$(grep -cP "\b${CODE}\b" "$TMPDIR/freqtrade_rag.log" || echo 0)
  [ "$COUNT" -gt 0 ] && printf "     HTTP %-4s → %s adet\n" "$CODE" "$COUNT"
done

echo ""
echo "  7b. Rate Limit & Failover Events:"
grep -ciE 'penalize|failover|rate.?limit|quota|too many|over capacity' "$TMPDIR/freqtrade_rag.log" 2>/dev/null \
  | xargs -I{} echo "     Toplam: {}"

echo ""
echo "  7c. Model Kullanım Dağılımı:"
grep -oP '(gemini[a-z0-9.\-]+|gpt[a-z0-9.\-]+|claude[a-z0-9.\-]+|kimi[a-z0-9.\-]+|llama[a-z0-9.\-]+|mixtral[a-z0-9.\-]+|deepseek[a-z0-9.\-]+|qwen[a-z0-9/.\-]+|groq[a-z0-9.\-]*)' "$TMPDIR/freqtrade_rag.log" 2>/dev/null \
  | sort | uniq -c | sort -rn | head -20 \
  | while IFS= read -r line; do echo "     $line"; done

echo ""
echo "  7d. Penalize edilen modeller (failover):"
grep -iE 'penalize|failover' "$TMPDIR/freqtrade_rag.log" 2>/dev/null \
  | grep -oP '(gemini[a-z0-9.\-/]+|qwen[a-z0-9.\-/]+|kimi[a-z0-9.\-]+|llama[a-z0-9.\-]+|deepseek[a-z0-9.\-]+|groq[a-z0-9.\-]*)' \
  | sort | uniq -c | sort -rn | head -10 \
  | while IFS= read -r line; do echo "     $line"; done

echo ""
echo "  7e. Son 20 LLM Hatası (timestamp ile):"
grep -iE '429|503|413|rate.?limit|quota|over capacity|penalize|failover|ServerError' "$TMPDIR/freqtrade_rag.log" 2>/dev/null \
  | tail -20 | while IFS= read -r line; do
      echo "     $(echo "$line" | cut -c1-180)"
    done
echo ""

########################################################################
# 8. TRADE & OPPORTUNITY ANALİZİ
########################################################################
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  8. TRADE & OPPORTUNITY ANALİZİ"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Freqtrade trade'leri AYRI DB'de: tradesv3.sqlite
TRADE_DB=~/freqtrade/user_data/tradesv3.sqlite
if [ -f "$TRADE_DB" ]; then
  echo ""
  echo "  8a. Açık Trade'ler (tradesv3.sqlite):"
  sqlite3 -header -column "$TRADE_DB" "
    SELECT id, pair, stake_amount, ROUND(open_rate,6) as open_rate,
           datetime(open_date,'localtime') as opened,
           is_short, leverage
    FROM trades WHERE is_open=1 ORDER BY open_date DESC;
  " 2>/dev/null || echo "     YOK veya HATA"

  echo ""
  echo "  8b. Son 30 Kapanan Trade:"
  sqlite3 -header -column "$TRADE_DB" "
    SELECT id, pair,
           ROUND(close_profit*100,2) as pnl_pct,
           ROUND(close_profit_abs,4) as pnl_abs,
           datetime(open_date,'localtime') as opened,
           datetime(close_date,'localtime') as closed,
           exit_reason
    FROM trades WHERE is_open=0 ORDER BY close_date DESC LIMIT 30;
  " 2>/dev/null || echo "     YOK veya HATA"

  echo ""
  echo "  8c. Trade Özet İstatistikleri:"
  sqlite3 -header -column "$TRADE_DB" "
    SELECT
      COUNT(*) as total,
      SUM(CASE WHEN is_open=1 THEN 1 ELSE 0 END) as open_tr,
      SUM(CASE WHEN is_open=0 THEN 1 ELSE 0 END) as closed_tr,
      ROUND(SUM(CASE WHEN is_open=0 THEN close_profit_abs ELSE 0 END),4) as total_pnl,
      ROUND(AVG(CASE WHEN is_open=0 THEN close_profit ELSE NULL END)*100,2) as avg_pct,
      SUM(CASE WHEN is_open=0 AND close_profit>0 THEN 1 ELSE 0 END) as wins,
      SUM(CASE WHEN is_open=0 AND close_profit<=0 THEN 1 ELSE 0 END) as losses
    FROM trades;
  " 2>/dev/null || echo "     HATA"

  echo ""
  echo "  8d. Günlük PnL:"
  sqlite3 -header -column "$TRADE_DB" "
    SELECT date(close_date) as day,
           COUNT(*) as trades,
           ROUND(SUM(close_profit_abs),4) as pnl,
           SUM(CASE WHEN close_profit>0 THEN 1 ELSE 0 END) as wins,
           SUM(CASE WHEN close_profit<=0 THEN 1 ELSE 0 END) as losses
    FROM trades WHERE is_open=0
    GROUP BY day ORDER BY day;
  " 2>/dev/null || echo "     HATA"

  echo ""
  echo "  8d2. Exit Reason Dağılımı:"
  sqlite3 -header -column "$TRADE_DB" "
    SELECT exit_reason, COUNT(*) as cnt,
           ROUND(AVG(close_profit)*100,2) as avg_pnl_pct,
           ROUND(SUM(close_profit_abs),4) as total_pnl
    FROM trades WHERE is_open=0
    GROUP BY exit_reason ORDER BY cnt DESC;
  " 2>/dev/null || echo "     HATA"
else
  echo "  Trade DB bulunamadı: $TRADE_DB"
  echo "  Alternatif aranıyor..."
  find ~/freqtrade/user_data -name "*.sqlite" -type f 2>/dev/null | while IFS= read -r f; do
    echo "    $f ($(du -h "$f" | cut -f1))"
  done
fi

if [ -f "$DB" ]; then
  echo ""
  echo "  8e. Opportunity Scores (son 30):"
  sqlite3 -header -column "$DB" "
    SELECT datetime(timestamp,'localtime') as ts, pair,
           ROUND(composite_score,1) as score, top_type,
           ROUND(momentum_score,1) as mom, ROUND(reversion_score,1) as rev,
           ROUND(funding_score,1) as fund
    FROM opportunity_scores ORDER BY timestamp DESC LIMIT 30;
  " 2>/dev/null || echo "     HATA"

  echo ""
  echo "  8f. Agent Performance:"
  AGENT_COUNT=$(sqlite3 "$DB" "SELECT COUNT(*) FROM agent_performance;" 2>/dev/null || echo "ERR")
  echo "     Kayıt sayısı: $AGENT_COUNT"
  if [ "$AGENT_COUNT" != "0" ] && [ "$AGENT_COUNT" != "ERR" ]; then
    echo ""
    echo "     Agent Type Bazında Özet:"
    sqlite3 -header -column "$DB" "
      SELECT agent_type, COUNT(*) as n,
             SUM(CASE WHEN was_correct THEN 1 ELSE 0 END) as correct,
             ROUND(AVG(outcome_pnl),3) as avg_pnl,
             ROUND(SUM(outcome_pnl),2) as total_pnl
      FROM agent_performance GROUP BY agent_type ORDER BY n DESC;
    " 2>/dev/null
    echo ""
    echo "     Son 20 kayıt:"
    sqlite3 -header -column "$DB" "SELECT * FROM agent_performance ORDER BY rowid DESC LIMIT 20;" 2>/dev/null
  fi

  echo ""
  echo "  8g. Agent Memory (son 20):"
  sqlite3 -header -column "$DB" "
    SELECT datetime(timestamp,'localtime') as ts, agent_type, pair, signal, ROUND(strength,3) as strength
    FROM agent_memory ORDER BY timestamp DESC LIMIT 20;
  " 2>/dev/null || echo "     BOŞ veya HATA"

  echo ""
  echo "  8h. Cross-Pair Cache:"
  sqlite3 -header -column "$DB" "
    SELECT datetime(timestamp,'localtime') as ts,
           substr(data_json,1,300) as data_preview
    FROM cross_pair_cache ORDER BY timestamp DESC LIMIT 5;
  " 2>/dev/null || echo "     BOŞ veya HATA"

  echo ""
  echo "  8i. Graduated Execution (Shadow vs Real):"
  echo "     REAL trade sinyalleri:"
  grep -c "Signal:REAL" "$TMPDIR/bot.log" 2>/dev/null || echo "     0"
  echo "     SHADOW sinyalleri:"
  grep -c "Signal:SHADOW" "$TMPDIR/bot.log" 2>/dev/null || echo "     0"
  echo "     SHADOW_WEAK sinyalleri:"
  grep -c "Signal:SHADOW_WEAK" "$TMPDIR/bot.log" 2>/dev/null || echo "     0"
  echo ""
  echo "     Son 20 SHADOW sinyal:"
  grep "Signal:SHADOW" "$TMPDIR/bot.log" 2>/dev/null | tail -20 | while IFS= read -r line; do
    echo "     $(echo "$line" | cut -c1-160)"
  done
  echo ""
  echo "     Son 10 REAL sinyal:"
  grep "Signal:REAL" "$TMPDIR/bot.log" 2>/dev/null | tail -10 | while IFS= read -r line; do
    echo "     $(echo "$line" | cut -c1-160)"
  done

  echo ""
  echo "  8j. Forgone P&L Özet (shadow trade sonuçları):"
  sqlite3 -header -column "$DB" "
    SELECT
      COUNT(*) as total,
      SUM(CASE WHEN was_executed=1 THEN 1 ELSE 0 END) as real_trades,
      SUM(CASE WHEN was_executed=0 THEN 1 ELSE 0 END) as shadow_trades,
      ROUND(AVG(CASE WHEN was_executed=0 AND forgone_pnl IS NOT NULL THEN forgone_pnl END),3) as avg_shadow_pnl,
      ROUND(AVG(CASE WHEN was_executed=1 AND forgone_pnl IS NOT NULL THEN forgone_pnl END),3) as avg_real_pnl,
      SUM(CASE WHEN was_executed=0 AND forgone_pnl>0 THEN 1 ELSE 0 END) as shadow_wins,
      SUM(CASE WHEN was_executed=0 AND forgone_pnl<=0 THEN 1 ELSE 0 END) as shadow_losses
    FROM forgone_profit
    WHERE signal_time > datetime('now','-7 days');
  " 2>/dev/null || echo "     HATA"

  echo ""
  echo "  8k. Forgone — Confidence Band Performansı:"
  sqlite3 -header -column "$DB" "
    SELECT
      CASE
        WHEN confidence < 0.30 THEN '[0-0.30) WEAK'
        WHEN confidence < 0.55 THEN '[0.30-0.55) SHADOW'
        WHEN confidence < 0.70 THEN '[0.55-0.70) REAL'
        ELSE '[0.70+) STRONG'
      END as band,
      COUNT(*) as cnt,
      ROUND(AVG(forgone_pnl),3) as avg_pnl,
      SUM(CASE WHEN forgone_pnl>0 THEN 1 ELSE 0 END) as wins,
      SUM(CASE WHEN forgone_pnl<=0 THEN 1 ELSE 0 END) as losses
    FROM forgone_profit
    WHERE forgone_pnl IS NOT NULL
    GROUP BY band ORDER BY band;
  " 2>/dev/null || echo "     HATA (henüz resolve edilmemiş olabilir)"
fi
echo ""

########################################################################
# 9. SCHEDULER ANALİZİ
########################################################################
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  9. SCHEDULER ANALİZİ"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "  9a. Job Çalışma Sayıları:"
for JOB in opportunity_scan agent_rebalance cross_pair_intel event_reanalysis \
           update_derivatives update_magma refresh_fear_greed bootstrap_pattern \
           health_check cleanup rolling_sentiment macro_update; do
  COUNT=$(grep -ci "$JOB" "$TMPDIR/freqtrade_scheduler.log" || echo 0)
  [ "$COUNT" -gt 0 ] && printf "     %-25s → %s\n" "$JOB" "$COUNT"
done

echo ""
echo "  9b. Scheduler Hataları (unique):"
grep -iE 'error|exception|failed|traceback' "$TMPDIR/freqtrade_scheduler.log" 2>/dev/null \
  | sed 's/^[0-9T:.Z+\- ]*//' | sort -u | head -30 \
  | while IFS= read -r line; do echo "     $(echo "$line" | cut -c1-150)"; done

echo ""
echo "  9c. numpy.matrix hatası detay:"
grep -i "numpy.*matrix\|np\.matrix\|matrix.*attribute" "$TMPDIR/freqtrade_scheduler.log" 2>/dev/null \
  | head -5 | while IFS= read -r line; do
      echo "     $(echo "$line" | cut -c1-180)"
    done

echo ""
echo "  9d. yfinance DXY hatası detay:"
grep -i 'DX=F\|dollar.*index\|yfinance.*error' "$TMPDIR/freqtrade_scheduler.log" 2>/dev/null \
  | head -5 | while IFS= read -r line; do
      echo "     $(echo "$line" | cut -c1-180)"
    done

echo ""
echo "  9e. Event Reanalysis tetiklemeleri:"
grep -i "event_reanalysis\|fear.*greed.*trigger\|funding.*trigger\|re.analyz" "$TMPDIR/freqtrade_scheduler.log" 2>/dev/null \
  | tail -15 | while IFS= read -r line; do
      echo "     $(echo "$line" | cut -c1-150)"
    done
echo ""

########################################################################
# 10. FD & RESOURCE
########################################################################
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  10. RESOURCE & FD ANALİZİ"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "  10a. Process FD Sayıları:"
for svc in freqtrade freqtrade-rag freqtrade-scheduler freqtrade-models; do
  PID=$(systemctl show "$svc" --property=MainPID --value 2>/dev/null || echo 0)
  if [ "$PID" != "0" ] && [ -d "/proc/$PID" ]; then
    FD_COUNT=$(ls /proc/$PID/fd 2>/dev/null | wc -l)
    FD_LIMIT=$(grep "Max open files" /proc/$PID/limits 2>/dev/null | awk '{print $4}')
    printf "     %-28s PID=%-8s FDs=%-6s Limit=%s\n" "$svc" "$PID" "$FD_COUNT" "$FD_LIMIT"
  else
    printf "     %-28s PID=%-8s (not running or PID=0)\n" "$svc" "$PID"
  fi
done

echo ""
echo "  10b. 'Too many open files' kontrol:"
TOTAL_FD_ERR=0
for f in "$TMPDIR"/*.log; do
  C=$(grep -ci "too many open files\|EMFILE\|fd leak" "$f" || echo 0)
  TOTAL_FD_ERR=$((TOTAL_FD_ERR + C))
  [ "$C" -gt 0 ] && echo "     $(basename "$f"): $C adet"
done
echo "     Toplam FD hatası: $TOTAL_FD_ERR"

echo ""
echo "  10c. Top memory tüketicileri:"
ps aux --sort=-%mem | head -11 | awk '{printf "     %-8s %-8s %5s%% %5s%% %s\n", $1, $2, $3, $4, $11}'
echo ""

########################################################################
# 11. SIGNAL HEALTH CANLI
########################################################################
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  11. CANLI SIGNAL-HEALTH SNAPSHOT"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
curl -s --connect-timeout 5 http://localhost:8891/signal-health 2>/dev/null \
  | python3 -m json.tool 2>/dev/null || echo "  UNREACHABLE"
echo ""

########################################################################
# 12. DB TABLOLARI
########################################################################
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  12. DB TABLO DURUMLARI"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
if [ -f "$DB" ]; then
  echo "  DB Boyutu: $(du -h "$DB" | cut -f1)"
  echo ""
  sqlite3 "$DB" "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;" 2>/dev/null \
  | while read TBL; do
    ROW_COUNT=$(sqlite3 "$DB" "SELECT COUNT(*) FROM \"$TBL\";" 2>/dev/null || echo "?")
    COLS=$(sqlite3 "$DB" "SELECT COUNT(*) FROM pragma_table_info('$TBL');" 2>/dev/null || echo "?")
    printf "     %-35s %4s cols  %8s rows\n" "$TBL" "$COLS" "$ROW_COUNT"
  done
fi
echo ""

########################################################################
# 13. MADAM PİPELİNE
########################################################################
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  13. MADAM PİPELİNE"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "  13a. Tier Kullanımı:"
for TIER in "Tier 1" "Tier 2" "Tier 2.5" "Tier 3" "Tier 3.5" "Tier 4" "EvidenceFirst" "evidence_first"; do
  COUNT=$(grep -ci "$TIER" "$TMPDIR/freqtrade_rag.log" || echo 0)
  [ "$COUNT" -gt 0 ] && printf "     %-20s → %s\n" "$TIER" "$COUNT"
done

echo ""
echo "  13b. Semaphore:"
grep -ciE 'semaphore|busy|concurrent.*madam' "$TMPDIR/freqtrade_rag.log" 2>/dev/null \
  | xargs -I{} echo "     Semaphore olayları: {}"

echo ""
echo "  13c. Cache Hit/Miss:"
HIT=$(grep -ci "cache.*hit\|cache_hit\|HIT" "$TMPDIR/freqtrade_rag.log" || echo 0)
MISS=$(grep -ci "cache.*miss\|cache_miss\|MISS\|no.*cache" "$TMPDIR/freqtrade_rag.log" || echo 0)
echo "     Cache HIT:  $HIT"
echo "     Cache MISS: $MISS"

echo ""
echo "  13d. MADAM ortalama latency (tahmini):"
grep -oP 'took \d+\.?\d*s|latency.*\d+\.?\d*|duration.*\d+\.?\d*' "$TMPDIR/freqtrade_rag.log" 2>/dev/null \
  | head -20 | while IFS= read -r line; do echo "     $line"; done
echo ""

########################################################################
# 14. MODEL SERVER
########################################################################
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  14. MODEL SERVER (8895)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
if [ -s "$TMPDIR/freqtrade_models.log" ]; then
  ERR=$(grep -ciE 'error|exception' "$TMPDIR/freqtrade_models.log" || echo 0)
  OOM=$(grep -ciE 'out of memory|oom|killed|memory' "$TMPDIR/freqtrade_models.log" || echo 0)
  echo "     Errors:    $ERR"
  echo "     OOM hints: $OOM"
  echo ""
  echo "     Restart timestamps (from traceback):"
  grep "Traceback" "$TMPDIR/freqtrade_models.log" 2>/dev/null \
    | grep -oP '\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}' \
    | sort -u | while IFS= read -r line; do echo "       $line"; done
  echo ""
  echo "     Son 30 error satırı:"
  grep -iE 'error|exception|critical' "$TMPDIR/freqtrade_models.log" 2>/dev/null \
    | tail -30 | while IFS= read -r line; do
        echo "     $(echo "$line" | cut -c1-180)"
      done
else
  echo "     Log boş veya yok"
fi
echo ""

########################################################################
# 15. DERIVATIVES & MAGMA & F&G
########################################################################
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  15. DERIVATIVES & MAGMA & FEAR/GREED"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
if [ -f "$DB" ]; then
  echo ""
  echo "  15a. Derivatives Data:"
  sqlite3 -header -column "$DB" "
    SELECT COUNT(*) as total,
           datetime(MAX(timestamp),'localtime') as last_update,
           COUNT(DISTINCT pair) as unique_pairs
    FROM derivatives_data;
  " 2>/dev/null || echo "     HATA"

  echo ""
  echo "  15b. MAGMA Edges:"
  sqlite3 -header -column "$DB" "
    SELECT graph_type, COUNT(*) as cnt
    FROM magma_edges GROUP BY graph_type ORDER BY cnt DESC;
  " 2>/dev/null || echo "     HATA"

  echo ""
  echo "  15c. Fear & Greed (son 10):"
  sqlite3 -header -column "$DB" "
    SELECT datetime(timestamp,'localtime') as ts, value
    FROM fear_and_greed ORDER BY timestamp DESC LIMIT 10;
  " 2>/dev/null || echo "     HATA"

  echo ""
  echo "  15d. Derivatives — son 5 pair güncellemesi:"
  sqlite3 -header -column "$DB" "
    SELECT pair, datetime(MAX(timestamp),'localtime') as last_update, COUNT(*) as entries
    FROM derivatives_data GROUP BY pair ORDER BY last_update DESC LIMIT 5;
  " 2>/dev/null || echo "     HATA"
fi
echo ""

########################################################################
# 16. SIGNAL_HEALTH TABLOSU
########################################################################
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  16. SIGNAL_HEALTH DB"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
if [ -f "$DB" ]; then
  echo "  Şema:"
  sqlite3 "$DB" "PRAGMA table_info(signal_health);" 2>/dev/null \
    | while IFS= read -r line; do echo "     $line"; done
  echo ""
  echo "  Son 30 kayıt:"
  sqlite3 -header -column "$DB" "SELECT * FROM signal_health ORDER BY timestamp DESC LIMIT 30;" 2>/dev/null
  echo ""
  echo "  Signal Type Dağılımı:"
  sqlite3 -header -column "$DB" "
    SELECT signal_type, COUNT(*) as cnt,
           ROUND(AVG(confidence),4) as avg_conf,
           ROUND(MIN(confidence),4) as min_conf,
           ROUND(MAX(confidence),4) as max_conf
    FROM signal_health GROUP BY signal_type ORDER BY cnt DESC;
  " 2>/dev/null || echo "     HATA"
fi
echo ""

########################################################################
# 17. FORGONE PnL
########################################################################
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  17. FORGONE P&L & MISSED TRADES"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
grep -iE 'forgone|missed|skip|rejected.*trade|not.*entering|below.*threshold' "$TMPDIR/bot.log" 2>/dev/null \
  | tail -30 | while IFS= read -r line; do
      echo "     $(echo "$line" | cut -c1-180)"
    done
echo ""

########################################################################
# 18. ÖZET
########################################################################
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  18. ÖZET SKOR KARTI"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
TOTAL_ERRS=0
for f in "$TMPDIR"/*.log; do
  C=$(grep -ciE 'error|exception|traceback|critical' "$f" || echo 0)
  TOTAL_ERRS=$((TOTAL_ERRS + C))
done
TOTAL_429=0
for f in "$TMPDIR"/*.log; do
  C=$(grep -cP '\b429\b' "$f" || echo 0)
  TOTAL_429=$((TOTAL_429 + C))
done
TOTAL_503=0
for f in "$TMPDIR"/*.log; do
  C=$(grep -cP '\b503\b' "$f" || echo 0)
  TOTAL_503=$((TOTAL_503 + C))
done
SERVICES_UP=$(systemctl is-active freqtrade freqtrade-rag freqtrade-scheduler freqtrade-models 2>/dev/null | grep -c "^active$" || echo 0)

echo "  ┌──────────────────────────────────────────────┐"
printf "  │  Servisler Ayakta:       %s / 4                 │\n" "$SERVICES_UP"
printf "  │  Toplam Error/Exception: %-20s │\n" "$TOTAL_ERRS"
printf "  │  HTTP 429 (Rate Limit):  %-20s │\n" "$TOTAL_429"
printf "  │  HTTP 503 (Unavailable): %-20s │\n" "$TOTAL_503"
printf "  │  FD Hatası (EMFILE):     %-20s │\n" "$TOTAL_FD_ERR"
printf "  │  Model Server Restarts:  %-20s │\n" "$(systemctl show freqtrade-models --property=NRestarts --value 2>/dev/null || echo '?')"
echo "  └──────────────────────────────────────────────┘"
echo ""

########################################################################
# 19. RAW SAMPLE — Son 200 satır (her servis)
########################################################################
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  19. RAW LOG SAMPLES (debug için)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
for LOGFILE in "$TMPDIR"/bot.log "$TMPDIR"/freqtrade_rag.log "$TMPDIR"/freqtrade_scheduler.log "$TMPDIR"/freqtrade_models.log; do
  BASENAME=$(basename "$LOGFILE" .log)
  [ ! -s "$LOGFILE" ] && continue
  echo ""
  echo "  ── $BASENAME (son 200 satır) ──"
  tail -200 "$LOGFILE" | while IFS= read -r line; do
    echo "  $line"
  done
done
echo ""

echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║  ANALİZ TAMAMLANDI — $(date '+%Y-%m-%d %H:%M:%S')                      ║"
echo "║  Dosya: $OUT                                                    ║"
echo "╚══════════════════════════════════════════════════════════════════╝"

# Dosya boyutu
sync
echo ""
echo "Dosya boyutu: $(du -h "$OUT" | cut -f1)"