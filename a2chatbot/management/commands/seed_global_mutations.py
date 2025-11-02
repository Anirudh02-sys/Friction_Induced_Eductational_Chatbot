from django.core.management.base import BaseCommand
from a2chatbot.vectorstore import get_collection, embed_text, chunk_text

class Command(BaseCommand):
    help = "Seeds mutation transcript into global vector store"

    def handle(self, *args, **options):
        coll = get_collection("global_mutation")

        with open("a2chatbot/seed_data/mutation.txt", "r") as f:
            text = f.read()

        chunks = chunk_text(text, chunk_size=300)
        embeddings = embed_text(chunks)
        ids = [f"global_{i}" for i in range(len(chunks))]

        # add to chroma
        coll.add(documents=chunks, embeddings=embeddings, ids=ids)

        self.stdout.write(self.style.SUCCESS("Global mutation knowledge seeded into Chroma"))
