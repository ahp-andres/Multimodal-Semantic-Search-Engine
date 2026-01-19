import os
import torch
import streamlit as st
import pandas as pd

from dotenv import load_dotenv
from qdrant_client import QdrantClient, models
from transformers import CLIPModel, CLIPProcessor
# ------------------------------
# Cargar catálogo local (CSV)
# ------------------------------
@st.cache_data
def load_inventory():
    # Ajusta la ruta si tu CSV está en otra carpeta
    return pd.read_csv("../dataset/clothing_segment_example.csv", sep="|", encoding="utf-8")

inventory_df = load_inventory()
# Nos aseguramos de que el id sea entero
inventory_df["id"] = inventory_df["id"].astype(int)


# ------------------------------
# Carga de variables de entorno
# ------------------------------
load_dotenv()  # lee .env

QDRANT_ENDPOINT = os.getenv("QDRANT_ENDPOINT")
QDRANT_KEY = os.getenv("QDRANT_KEY")

# ------------------------------
# Clientes y modelos (cacheados)
# ------------------------------
@st.cache_resource
def get_qdrant_client():
    return QdrantClient(
        url=QDRANT_ENDPOINT,
        api_key=QDRANT_KEY,
        prefer_grpc=True,
    )

@st.cache_resource
def load_clip():
    model_name = "laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K"
    model = CLIPModel.from_pretrained(model_name)
    processor = CLIPProcessor.from_pretrained(model_name)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    return model, processor, device

def get_text_embedding(text: str):
    model, processor, device = load_clip()
    inputs = processor(
        text=[text],
        images=None,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )

    # solo usamos las llaves de texto
    inputs = {k: v.to(device) for k, v in inputs.items() if k in ["input_ids", "attention_mask"]}

    with torch.no_grad():
        text_features = model.get_text_features(**inputs)
        # normalizamos (opcional, pero suele ayudar)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    # devolvemos como lista de floats (Qdrant lo espera así)
    return text_features.cpu().numpy()[0].tolist()

def search_products(query_text: str,
                    brand: str | None,
                    product_type: str | None,
                    top_k: int = 10):
    client = get_qdrant_client()
    query_vector = get_text_embedding(query_text)

    # ---- armamos las condiciones de filtro (igual que en el notebook) ----
    filter_conditions: list[models.FieldCondition] = []

    if brand:
        filter_conditions.append(
            models.FieldCondition(
                key="brand",
                match=models.MatchValue(value=brand)
            )
        )

    if product_type and product_type != "Todos":
        filter_conditions.append(
            models.FieldCondition(
                key="product_type",
                match=models.MatchValue(value=product_type)
            )
        )

    payload_filter = (
        models.Filter(must=filter_conditions)
        if filter_conditions
        else None
    )

    # ---- consulta a Qdrant ----
    response = client.query_points(
        collection_name="myfashionproducts",
        query=query_vector,
        query_filter=payload_filter,   # 👈 aquí entra el filtro
        with_payload=True,
        with_vectors=False,            # no necesitamos el vector en la respuesta
        limit=top_k,
    )

    return response

# ---------------------------------
# Interfaz Web
# ---------------------------------
st.set_page_config(page_title="Búsqueda Semántica de Moda", layout="wide")

st.title("🔍 Búsqueda Semántica de Productos de Moda")
st.markdown(
    """
    Explora el catálogo como si fuera una tienda online inteligente 🛍️  
    Escribe una descripción en lenguaje natural y, opcionalmente, filtra por marca o tipo de producto.
    """
)

col_q, col_brand, col_type = st.columns([3, 1.5, 1.5])

with col_q:
    query_text = st.text_input(
        "Búsqueda",
        placeholder="Ejemplo: 'vestido blanco bohemio para verano'",
    )

with col_brand:
    brand = st.text_input(
        "Marca (opcional)",
        placeholder="Ejemplo: 'Adicora'",
    )

with col_type:
    product_type = st.selectbox(
        "Tipo de producto (opcional)",
        options=["Todos",
                 "Women > Clothing > Dresses",
                 "Women > Clothing > Swimwear",
                 "Women > Clothing > Trousers",
                 "Women > Clothing > Tops"],
    )

top_k = st.slider("Número de resultados", min_value=1, max_value=20, value=8)

# 👇👇 TODO EL MANEJO DE 'result' VA DENTRO DEL BOTÓN
if st.button("Buscar") and query_text.strip():
    st.write("---")
    st.subheader("Resultados")

    # 1) Hacemos la consulta
    with st.spinner("Consultando la base vectorial..."):
        result = search_products(
            query_text=query_text.strip(),
            brand=brand.strip() or None,
            product_type=None if product_type == "Todos" else product_type,
            top_k=top_k,
        )

    # 2) Revisamos si hay puntos
    if not result.points:
        st.warning("No se encontraron productos que coincidan con tu búsqueda y filtros.")
    else:
        n_cols = 4
        cols = st.columns(n_cols)

        for i, point in enumerate(result.points):
            c = cols[i % n_cols]
            payload = point.payload or {}

            # --- Datos básicos del payload (Qdrant) ---
            brand_val = payload.get("brand", "N/A")
            ptype_val = payload.get("product_type", "N/A")

            # --- Buscamos la fila en el CSV usando el id ---
            row = inventory_df[inventory_df["id"] == point.id]

            if not row.empty:
                name = row["product_name"].iloc[0]
                img_url = row["product_image"].iloc[0]
            else:
                name = "Producto sin nombre"
                img_url = None

            with c:
                # Imagen
                if img_url and isinstance(img_url, str) and img_url.strip():
                    st.image(img_url, caption=name, use_container_width=True)
                else:
                    st.write("(Sin imagen)")

                # Texto
                st.markdown(f"**Marca:** {brand_val}")
                st.markdown(f"**Tipo de producto:** {ptype_val}")
                st.markdown(f"**Score:** {round(point.score, 4)}")

