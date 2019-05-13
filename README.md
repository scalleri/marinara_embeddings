# n\_hot\_embeddings

# Wordembeddings and NGrams
Das ist eine eher allgemeine Erklärung für die verschiedenen Pythonscripts, unter `docs/` hat es noch zu den projektspeziefischeren Experimenten Erklärungen. 

## Wordembeddings berechnen

Um normale Wordembeddings auf Tokenebene ohne POS Informationen oder Phrasen zu berechnen, kann das Script `compute_models.py` verwendet werden.

Das Script kann entweder die Token direkt aus ES, der CWB oder einem tokenisierten Text (ein Token pro Zeile) berechnen. 

Die Funktion für die WE-Berchnung ist folgende:

```
model = Word2Vec(texts,window=6,sorted_vocab=1,max_final_vocab=None,min_count=10,sample=1e-5,sg=1,workers=15)
```

Je nach Anwendungsfall müssen die Parameter angepasst werden, genauere Infos zu den Parametern: https://radimrehurek.com/gensim/models/word2vec.html#gensim.models.word2vec.Word2Vec.trainables


### Elastic Search Anbindung

Der Code berechnet Wordembeddings und ruft den code `get_data.py` auf um Daten aus Elasticsearch zu ziehen.

Der Code kann mehrere ES Abfragen machen, die key/value-Paare müssen beim Aufrufen einfach definiert werden:

```
pyhon3 compute_models.py -k source source source -v admin 20min blick -mn admin_20min_blick 
```

Berechnet das WE-Modell für admin, 20min und blick zusammen und speichert das Modell unter `embedding_models/admin_20min_blick.model`

Für andere Parameter siehe -h Funktion.

#### Lemmata aus Elastic Search lesen

`get_data.py`

Is the code that reads the specified data (lemma, ne, pos, words ... from the esclient)

### CWB Anbindung 
Stream von Lemma mit `<text>` Attributen aus der CWB lesen 
```
$cwb-decode -Cx [CWB-Korpusname] -S text -P lemma
```

Damit die gewünschten Tokens direkt in das `compute_models.py` script gelesen werden können:

```
$cwb-decode -Cx [CWB-Korpusname] -S text -P lemma | python3 compute_models.py -cwb True -mn cwb_test
```

Wenn `-cwb` auf True gesetzt ist, bedeuted dass, das der Input ein Tokenstream aus der CWB ist, die `-s`-flag bezieht sich auf das s-attribut, welches die Tokens auftrennt, default ist _text_ aber es kann bei `-s` definiert werden. 


### Tokenisiertes Textfile

```
python3 compute_models.py -f [FILENAME]
```

Das File besteht aus einem Lemma/Wort pro Zeile. 

## Andere Modelltypen

### Modelle mit POS Annotationen rechnen

Script: `compute_pos_models.py`
Berechnet WE nicht auf dem reinen Lemma aber auf Lemma_pos -> disambiguierung und einfachere Abfrage von Wortarten bei der Analyse von WE 


### WE Modell mit Phrasen (ngrams) berechnen
-> Damit Artikelnamen etc. als 1 Token gesehen werden, Mehrwortausdrücke (MWA) werden so einfacher identifiziert

Dieser approach eignet sich sehr v.a. für romanische Sprachen, welche häufig MWA (Mehrwortausdrücke) haben. MWA lassen sich über Kookurrenzen herausfinden und gensim hat eine 'Phrase-Funktion', welche bigramme berechnen kann und die Wörter im Modell mit einem gewünschten Zeichen verbindet. Der wiederholte Aufruf der Funktion ermöglicht es ngramme zu finden und jeweils zusammenzufügen. Im Moment berechnet der Code bi und trigramme, aber wenn zwei bigramme aufeinanderfolgend signifikant sind werden auch 4gramme gebildet. 

```
# mit -P word, wenn mal Zahlen etc. beibehalten will sonst mit -P lemma  
$ cwb-decode -Cx [CWB-Korpusname] -S text -P lemma | python3 compute_phrase_models.py -cwb True -mn bbl_bge_all_de_trigram -max None

```

Folgender Teil des Codes berechnet ngramme:

```
phrases = Phrases(docs,min_count=5,threshold=10,delimiter=b"_")
bigram_phraser = Phraser(phrases)
phrased_texts = {}
for i in range(0,len(texts)-1):
    tokens = bigram_phraser[texts[ids[i]]]
    phrased_texts[ids[i]] = tokens 

phr_docs = [doc for id_,doc in sorted(phrased_texts.items())]
tri_phrases = Phrases(phr_docs,min_count=5,threshold=5,delimiter=b"_")
trigram_pharser = Phraser(tri_phrases)

tri_phrased_texts = {}
count = 1
for i in range(0,len(texts)-1):
    tokens = trigram_pharser[phrased_texts[ids[i]]]
    tri_phrased_texts[ids[i]]=tokens

```

Die Funktion `Phrases()` setzt die Parameter für die 'Erkennung' von Phrasen, also in welchen Texten (hier `docs`), Mindestvorkommnis der Phrase (`min_count=5`), der Mindestwert der Phrasenberechnung (eine Art Score, der die Signifikanz der Phrase bewertet, wenn es nicht ein NPMI scoring ist, dann ist ein Score zwischen 5 und 15 sinnvoll vgl. https://arxiv.org/abs/1310.4546 , NPMI geht nur von -1 bis 1, vgl. https://svn.spraakdata.gu.se/repos/gerlof/pub/www/Docs/npmi-pfd.pdf ), die entdeckten Phrasen werden mit dem Delimiter-Zeichen verbunden (hier `delimiter=b'_'`), welches als byte encoded werden muss, darum das _b_ vor dem String.

Nachdem die Phrasen berechnet wurden, werden sie mit der `Phraser()`-Funktion angewendet. Dann iteriert man über den Text und es wird ein dictionary generiert in der Form {textid:"text"}, in einem ersten Schritt für bigramme. 
Um auch längere Phrasen zu erkennen, wird ein neues Phrases Modell berechnet, mit einem nicht so hohen threshold (kann je nach Anwendungsfall angepasst werden), tendenziell sind aber längere Phrasen weniger signifikant, da sie weniger oft vorkommen. 

Das neu generierte Korpus mit den markierten Phrasen wird als JSON file gespeichert und ein WE modell wird auf dem Text inkl. markierte Phrasen berechnet. 

#### Textbeispiel trigram preprocessing
```
['Urteilskopf_88', 'IV123', '33', '.', 'Auszug_aus_dem', 'Urteil_des_Kassationshofes', 'vom_26', '.', 'Oktober_1962_i.S.', 'Schlüchter', 'gegen', 'Schrag', 'und', 'Verlag', 'des', 'Schweiz.', 'Kaufmännischen_Vereins', 'sowie', 'Staatsanwaltschaft_des_Kantons', 'Zürlch', '.', 'Regeste_Art.', '1', 'Abs._2', ',', 'Art._42_Ziff._1', 'lit._a', 'und', 'b', 'URG', '.', '1', '.', 'Das', 'Urheberrecht_an', 'einem', 'Lehrbuch', 'ist', 'auch', 'verletzt', ',', 'wenn', 'das_Werk', 'in', 'seinen_charakteristischen', 'Grundzügen', ',', 'namentlich_hinsichtlich', 'Planung', ',', 'Auswahl', 'und', 'Erfassen', 'des_Stoffes', 'oder', 'Anordnung', 'und', 'Gliederung', 'desselben', ',', 'übernommen', 'wird', '(_Erw.', '1', ')', '.', '2', '.', 'An', 'Übungen', 'und', 'Anleitungen', 'eines', 'Lehrbuches', 'für', 'Maschinenschreiben', 'besteht', 'Urheberrecht', ',', 'wenn_sie', 'originelles', 'Ergebnis', 'geistigen_Schaffens', 'sind', '(_Erw._2', ')', '.', '3', '.', 'Zum_Verhältnis', 'von', 'Art._42_Ziff._1', 'lit._a', 'zu', '43_Ziff.', '2_URG', '(_Erw._3', ')', '.', '4', '.', 'Mit', 'Werk_im_Sinne', 'des', 'Art._42_Ziff._1', 'lit._b', 'URG', 'ist', 'nicht', 'das', 'wiedergebende', ',', 'sondern', 'das', 'wiedergegebene', 'gemeint', '(_Erw._4', ')', '.', 'Sachverhalt_ab_Seite', '124', 'A._-', 'Fritz', 'Schrag', ',', 'Lehrer', 'an', 'der', 'Kantonalen', 'Handelsschule', 'in', 'Zürich', ',', 'verfasste', 'ein', 'Lehrbuch', 'für', 'Maschinenschreiben', ',', 'das', '1958', 'im', 'Verlag', 'des_Schweizerischen_Kaufmännischen', 'Vereins', 'in', 'neunter', 'Auflage', 'erschien', '.', 'Das', 'Kaufmännische', 'Lehrinstitut', 'in', 'Zürich', 'erteilte', 'unter', 'der', 'Leitung', 'von', 'Hans', 'Schlüchter', 'Fernkurse', 'für', 'Maschinenschreiben', ',', 'wobei', 'den', 'Schülern', 'fünf', 'Lehrhefte', 'zugestellt_wurden', '.', 'Das', 'letzte', 'Heft', ',', 'welches', 'die', 'Seiten', '55', 'bis', '71', 'des', 'Lehrganges', 'umfasste', ',', 'wurde', 'von', 'Heidi_Schlüchter', 'zusammengestellt', ',', 'vervielfältigt', 'und', 'bis', '1', '.', 'April_1958', 'an', 'ungefähr', 'zwanzig', 'Schüler', 'versandt', '.', 'Schrag', 'hielt', 'den', 'Lehrgang', 'Schlüchters', 'für', 'ein', 'Plagiat', 'seines', 'Lehrbuches', 'und', 'stellte', 'Strafantrag_wegen_Verletzung', 'von_Urheberrechten', '.', 'B._-', 'Das_Obergericht_des_Kantons', 'Zürich', 'erklärte', 'als_Berufungsinstanz', 'am', '4', '.', 'Mai_1961', 'Heidi_Schlüchter', 'der', 'Übertretung', 'von', 'Art._42_Ziff._1', 'lit._a', 'und', 'b', 'URG', 'schuldig', 'und', 'verurteilte_sie', 'zu', 'einer_bedingt_vorzeitig', 'löschbaren_Busse_von', 'Fr._150', '.', '-.', 'Den', 'Werkcharakter', 'des', 'Lehrbuches', 'von', 'Schrag', 'bejahte', 'es', 'vor_allem', 'aus_folgenden_Gründen', ':', 'Entscheidend_sei', ',', 'dass', 'über', 'ein_blosses', 'quantitatives', 'Zusammenstellen', 'eines_Stoffes', 'hinaus', 'eine', 'qualitative', 'und', 'einigermassen', 'originelle', 'wissenschaftliche_Bearbeitung', 'stattgefunden_habe', '.', 'Freilich', 'habe', 'die', 'Gebundenheit', 'an', 'die', 'Tastaturanordnung', ',', 'an', 'das', 'Zehnfingersystem', ',', 'an', 'die', 'Schrift', 'und', 'auch', 'an', 'den', 'Stoff', 'dem', 'individuell-schöpferischen', 'Schaffen', 'keinen', 'grossen_Spielraum', 'gelassen', ';', 'dennoch', 'sei', 'unverkennbar', ',', 
```



## Tensorboard 

### Tensorboard generieren

Mit dem Code `visualize_we.py` können die berechneten Modelle gleichzeitig visualisiert werden. Der Code greift auf alle vorberechneten Modelle im spezifizierten Ordner (default 'embedding\_models') zu und wandelt sie in einen Tensorboard-Graphen um. Alle nötigen files werden, wenn nicht anderst definiert im Ordner 'tensorboard_test' abgespeichert. Es kann aber auch ein anderer Zielordner definiert werden. 

### Visualisierung starten
Damit Tensorboard auf die Modelle zugreifen kann muss die Visualisierung gestartet werden.
Mit dem Befehl:
```
tensorboard --logdir="tensorboard_test/" --port 8083
```

kann die Visualisierung gestartet werden. 

Idealerweise startet man die Visualisierung in nohup, damit das tensorboard aktiv bleibt auch wenn man sich ausloggt:

```
$ nohup tensorboard --logdir="tensorboard_test/" --port 8083 &
```



### Mehrere Modelle in den gleichen Raum projezieren

Das Script `multi_visualize.py` ermöglicht es mehrere WE-Modelle in den gleichen Raum zu projezieren. Alle Modelle müssen im gleichen Ordner abgelegt sein. Die maximale Grösse ist auf 100'000 Wörter beschränkt, da sonst die Visualisierung nicht funktioniert. Damit alle Modelle gleich vorhanden sind werden die top n (n = 100k/Anzahl Modelle) Wörter aus dem Vocab jeweils visualisiert. Die Wörter bekommen jeweils ein Label mit dem Korpus/Modell dem sie angehören. 

-> Funktioniert, ist aber nicht wirklich sinnvoll 

### Modelle vergleichen Wort für Wort

Das Script haben wir für die Rechtspopulismus Analyse gebraucht. 
Script: `compare_models.py`
-> berechnet pro Wort in Modell 1 die cosine_sim zum gleichen Wort in Modell2

vgl. `docs/populismus experiments.md` für mehr Infos. 


### Wordembeddings abfragen 

Im Terminal den python3 interpreter starten: (das $-Zeichen zeigt eine Terminal eingabe an, muss nicht abgetippt werden)
Die meisten Embeddingmodelle liegen unter: `/home/call/n_hot_embeddings/embedding_models/`

Voraussetzung ist die Installation von gensim (https://radimrehurek.com/gensim/install.html) (normalerweise: pip3 install gensim/pip install gensim)
```
$ python3
[infos zu python3]
>>> from gensim.models.keyedvectors import KeyedVectors
>>> model = KeyedVectors.load('[MODELNAME.model]')
>>> model.most_similar('Digitalisierung')
[('Langzeitarchivierung', 0.8689799904823303), ('digital', 0.840603768825531), ('Bildplatte', 0.8246514797210693), ('multimedialen', 0.8082578182220459), ('Sammelauftrag', 0.8074219822883606), ('filmisch', 0.8007779717445374), ('Fernsehfilm', 0.8005862236022949), ('Computerspiel', 0.7991310358047485), ('Radiobereich', 0.7987841367721558), ('Spartenprogramm', 0.794711709022522)]

-> Zeigt die top 10 NN von 'Digitalisierung' an, inkl. Cosinus Distanz 

>>> model.most_similar('Person',topn=100)
# topn=INT -> anzahl der NN, die man sich anzeigenlassen will. 

um den Interprerter zu beenden:
>>> exit()
```
Unter https://radimrehurek.com/gensim/models/keyedvectors.html hat es gute Beispiele für andere mögliche Abfragen mit gensim 



## GermaNet Anbindung

Die GermaNet DB ist über die Python Library `pygermanet` abfragbar und kann folgendermassen importiert werden:

```
>>> from pygermanet import load_germanet
>>> gn = load_germanet()
```

GermaNet basiert auf der Wordnet-Struktur (siehe [WordNet](https://wordnet.princeton.edu)) und ist aus Synsets aufgebaut. Die Synsets bilden synonimische Beziehungen ab. Diese Synsets sind mit anderen über 'conceptual relations' verbunden. Ein Beispiel wie Hypo- und Hyperonyme abgefragt werden:

```
>>> gn.synsets('Energie') 
[Synset(Energie.n.1), Synset(Tatkraft.n.1)]

\# damit die spezifischen Hypero-/Hyponyme abgefragt werden können muss ein jeweiliges Synset von Energie, in diesem Beispiel 'Energie.n.1' ausgewählt werden. Der Einfachheit halber wird es der Variabel 'energie' zugewiesen.

>>> energie = gn.synset('Energie.n.1')
>>> energie.hyponyms
[Synset(Höhenenergie.n.1), Synset(Primärenergie.n.1), Synset(Heizenergie.n.1), Synset(Wärme.n.2), Synset(Haftenergie.n.1), Synset(Reibungsenergie.n.1), Synset(Elektrizität.n.1), Synset(Bewegungsenergie.n.1), Synset(Alternativenergie.n.1), Synset(Arbeit.n.1), Synset(Energiespektrum.n.1), Synset(Bindungsenergie.n.1), Synset(Lebensenergie.n.1), Synset(Aktivierungsenergie.n.1), Synset(Gesamtenergie.n.1), Synset(Strahlungsenergie.n.1), Synset(Energieniveau.n.1), Synset(Lichtenergie.n.1), Synset(Antriebsenergie.n.1), Synset(Schallenergie.n.1), Synset(Vakuumenergie.n.1), Synset(Reliefenergie.n.1), Synset(Strömungsenergie.n.1), Synset(Oberflächenenergie.n.1), Synset(Ruheenergie.n.1)]
>>> energie.hypernym_paths
[[Synset(GNROOT.n.1), Synset(Zustand.n.1), Synset(Situation.n.1), Synset(Ereignis.n.1), Synset(Erscheinung.n.2), Synset(natürliches Phänomen.n.1), Synset(physikalisches Phänomen.n.1), Synset(Energie.n.1)]]

# zeigt alle Relationen des Synsets an. 
>>> energie.rels()

```

