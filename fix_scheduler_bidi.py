import sys

file_path = "user_data/scripts/scheduler.py"

with open(file_path, "r") as f:
    content = f.read()

import_target = """from magma_memory import MAGMAMemory"""

import_replacement = """from magma_memory import MAGMAMemory
from bidirectional_rag import BidirectionalRAG"""

content = content.replace(import_target, import_replacement, 1)

cron_target = """    def _prune_magma_memory(self):
        \"\"\"Job: Clean up old/weak linkages inside MAGMA memory tables.\"\"\"
        logger.info("[Scheduler:Job] Pruning MAGMAMemory edges...")
        try:
            magma = MAGMAMemory()
            deleted = magma.prune(min_weight=0.5, max_age_days=180)
            logger.info(f"[Scheduler:Job] Removed {deleted} MAGMA connections.")
        except Exception as e:
            logger.error(f"[Scheduler:Job] MAGMAMemory pruning failed: {e}")"""

cron_replacement = """    def _embed_bidi_lessons(self):
        \"\"\"Job: Write back AI trade evaluation lessons into Vector DB.\"\"\"
        logger.info("[Scheduler:Job] Embedding Bidirectional RAG lessons...")
        try:
            bidi_rag = BidirectionalRAG()
            lessons = bidi_rag.get_unembedded_lessons()
            if not lessons:
                return

            # Push to VectorDB using same retriever
            from hybrid_retriever import HybridRetriever
            retriever = HybridRetriever(collection_name="crypto_news")
            
            docs, metas, ids = [], [], []
            for l in lessons:
                docs.append(l['lesson_text'])
                metas.append({
                    "type": "ai_lesson",
                    "pair": l['pair'],
                    "source": "bidirectional_rag",
                    "signal": l['signal'],
                    "outcome_pnl": float(l['outcome_pnl'])
                })
                ids.append(f"lesson_{l['id']}")
                
            retriever.add_documents(documents=docs, metadatas=metas, ids=ids)
            
            # Mark as embedded
            bidi_rag.mark_lessons_embedded([l['id'] for l in lessons])
            logger.info(f"[Scheduler:Job] Successfully integrated {len(lessons)} Bidirectional lessons.")
            
        except Exception as e:
            logger.error(f"[Scheduler:Job] Bidirectional embedding failed: {e}")

    def _prune_magma_memory(self):
        \"\"\"Job: Clean up old/weak linkages inside MAGMA memory tables.\"\"\"
        logger.info("[Scheduler:Job] Pruning MAGMAMemory edges...")
        try:
            magma = MAGMAMemory()
            deleted = magma.prune(min_weight=0.5, max_age_days=180)
            logger.info(f"[Scheduler:Job] Removed {deleted} MAGMA connections.")
        except Exception as e:
            logger.error(f"[Scheduler:Job] MAGMAMemory pruning failed: {e}")"""

content = content.replace(cron_target, cron_replacement, 1)

job_target = """        self.scheduler.add_job(
            self._prune_magma_memory,
            trigger=CronTrigger(day_of_week='sun', hour=3, minute=0, timezone=pytz.UTC),
            id='job_prune_magma_memory',
            name='Prune old/weak linkages inside MAGMA graphs',
            replace_existing=True
        )"""

job_replacement = """        self.scheduler.add_job(
            self._prune_magma_memory,
            trigger=CronTrigger(day_of_week='sun', hour=3, minute=0, timezone=pytz.UTC),
            id='job_prune_magma_memory',
            name='Prune old/weak linkages inside MAGMA graphs',
            replace_existing=True
        )

        # 3.2. AI Trade Lessons Processing (Nightly)
        self.scheduler.add_job(
            self._embed_bidi_lessons,
            trigger=CronTrigger(hour=4, minute=0, timezone=pytz.UTC),
            id='job_embed_bidi_lessons',
            name='Embed active trade evaluation lessons into Hybrid Vector memory',
            replace_existing=True
        )"""

content = content.replace(job_target, job_replacement, 1)

with open(file_path, "w") as f:
    f.write(content)
