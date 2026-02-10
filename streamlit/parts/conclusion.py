# conclusion du projet en reliant au maximum les résultats obtenus à la problématique métier
# critique et perspectives (ce qui aurait pû être fait avec plus de temps)
# par exemple : on aurait pu lire des articles sur des sujets similaires pour s'inspirer de leurs architectures

import streamlit as st


def show_conclusion():
    st.title("🎓 Conclusion & perspectives")
    st.header("🚧 Section en cours de développement")
    # st.markdown("""
    # - liste markdown
    # - item
    # """)

    # Screenshot
    # st.image("screenshots/contexte.png", caption="Vue d'ensemble")

    # Tableau
    # st.header("Header")
    # col1, col2 = st.columns(2)
    # with col1:
    #     st.metric("Catégories", "27")
    # with col2:
    #     st.metric("Articles", "~100K")


# Si exécuté directement (pour tester)
if __name__ == "__main__":
    st.set_page_config(layout="wide")
    show_conclusion()
