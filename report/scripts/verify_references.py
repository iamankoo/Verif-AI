"""Fetch the publisher/arXiv metadata of every reference from Crossref / arXiv and print it next to the bibliography entry,
so titles, authors, years and venues can be checked against the primary record.   python report/scripts/verify_references.py"""
import re, sys, json
import requests

DOI = {
    "karras2019stylegan": "10.1109/CVPR.2019.00453", "karras2020stylegan2": "10.1109/CVPR42600.2020.00813", "rossler2019ff": "10.1109/ICCV.2019.00009",
    "rombach2022ldm": "10.1109/CVPR52688.2022.01042", "verdoliva2020media": "10.1109/JSTSP.2020.3002101", "tolosana2020deepfakes": "10.1016/j.inffus.2020.06.014",
    "wang2020cnn": "10.1109/CVPR42600.2020.00872", "ojha2023universal": "10.1109/CVPR52729.2023.02345", "corvi2023diffusion": "10.1109/ICASSP49357.2023.10095167",
    "geirhos2020shortcut": "10.1038/s42256-020-00257-z", "afchar2018mesonet": "10.1109/WIFS.2018.8630761", "chollet2017xception": "10.1109/CVPR.2017.195",
    "he2016resnet": "10.1109/CVPR.2016.90", "fridrich2012srm": "10.1109/TIFS.2012.2190402", "selvaraju2017gradcam": "10.1109/ICCV.2017.74",
    "zhang2016mtcnn": "10.1109/LSP.2016.2603342", "li2020celebdf": "10.1109/CVPR42600.2020.00327", "zhou2017twostream": "10.1109/CVPRW.2017.229",
    "bayar2016constrained": "10.1145/2909827.2930786", "dang2020detection": "10.1109/CVPR42600.2020.00582", "rothe2018imdbwiki": "10.1007/s11263-016-0940-3",
    "wilson1927": "10.1080/01621459.1927.10502953",
    # candidates for the new novelty section
    "guera2018rnn": "10.1109/AVSS.2018.8639163", "li2020facexray": "10.1109/CVPR42600.2020.00505", "luo2021highfreq": "10.1109/CVPR46437.2021.01605",
    "qian2020f3net": "10.1007/978-3-030-58610-2_6", "masi2020twobranch": "10.1007/978-3-030-58571-6_39", "selvaraju2020gradcam_ijcv": "10.1007/s11263-019-01228-7",
    "tan_effnet_icml": None,
}
ARXIV = {"goodfellow2014gan": "1406.2661", "ho2020ddpm": "2006.11239", "dolhansky2020dfdc": "2006.07397", "song2023dff": "2309.02218",
         "tan2019efficientnet": "1905.11946", "guo2017calibration": "1706.04599", "paszke2019pytorch": "1912.01703", "loshchilov2019adamw": "1711.05101"}
only = set(sys.argv[1:])
for k, d in DOI.items():
    if not d or (only and k not in only):
        continue
    try:
        r = requests.get("https://api.crossref.org/works/" + d, timeout=30, headers={"User-Agent": "verifai-report-check (mailto:none)"})
        m = r.json()["message"]
        au = "; ".join(f"{a.get('family','?')}, {a.get('given','')[:1]}" for a in m.get("author", [])[:5]) + (" ..." if len(m.get("author", [])) > 5 else "")
        yr = (m.get("issued", {}).get("date-parts") or [[None]])[0][0]
        print(f"{k:24s} {d}\n   title: {m['title'][0]}\n   authors: {au}\n   year: {yr} | venue: {(m.get('container-title') or ['-'])[0]} | pages: {m.get('page','-')} vol/iss: {m.get('volume','-')}/{m.get('issue','-')}")
    except Exception as e:
        print(f"{k:24s} {d}  ERROR {e}")
for k, a in ARXIV.items():
    if only and k not in only:
        continue
    try:
        t = requests.get("http://export.arxiv.org/api/query?id_list=" + a, timeout=30).text
        title = re.search(r"<title>(.*?)</title>", t.split("<entry>")[1], re.S).group(1).strip().replace("\n", " ")
        auth = re.findall(r"<name>(.*?)</name>", t)[:6]
        yr = re.search(r"<published>(\d{4})", t).group(1)
        print(f"{k:24s} arXiv:{a}\n   title: {title}\n   authors: {'; '.join(auth)}\n   year(first version): {yr}")
    except Exception as e:
        print(f"{k:24s} arXiv:{a}  ERROR {e}")
