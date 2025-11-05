KIT (sem rótulos) — rodar com Python 3.9+
Dependências: pip install pillow numpy scikit-learn

1) Extrair features e manifesto:
   python features.py --root <pasta_dataset>

   Espera que suas imagens estejam em <root>/images/<PDF_BASE>/*.jpg|png
   Gera: <root>/metadata/features.npy e manifest.csv

2) Clusterizar vistas (ex.: 3 clusters por PDF_BASE):
   python cluster_views.py --root <pasta_dataset> --k 3 --copy
   Gera: metadata/clusters.csv e, se --copy, clusters/<PDF_BASE>/cluster_i/

3) Busca por similaridade (gera HTML):
   python nn_search.py --root <pasta_dataset> --query <caminho/da/imagem.jpg>
   Abre metadata/nn_search.html no navegador.

4) Galeria por voluntário (HTML):
   python make_gallery.py --root <pasta_dataset>
   Abre metadata/gallery.html no navegador.

Sugestões:
- Use os clusters como "tipos de vista" e corrija manualmente (labels fracos).
- Use nn_search para achar casos parecidos e discutir com avaliadores.
