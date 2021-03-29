# dance-dance-transformation
DDR simfile generator using a transformer architecture


# Building dataset

1. Make a directory named `data` under `~/ddc` (or change `scripts/var.sh` to point to a different directory)
1. Under `data`, make directories `raw`, `json_raw` and `json_filt`
1. Under `data/raw`, make directories `fraxtil` and `itg`
1. Under `data/raw/fraxil`, download and unzip:
    * [(Fraxtil) Tsunamix III](https://fra.xtil.net/simfiles/data/tsunamix/III/Tsunamix%20III%20[SM5].zip)
    * [(Fraxtil) Fraxtil's Arrow Arrangements](https://fra.xtil.net/simfiles/data/arrowarrangements/Fraxtil's%20Arrow%20Arrangements%20[SM5].zip)
    * [(Fraxtil) Fraxtil's Beast Beats](https://fra.xtil.net/simfiles/data/beastbeats/Fraxtil's%20Beast%20Beats%20[SM5].zip)
1. Under `data/raw/jubo`, download and unzip:
    * [(Jubo) 1-150](https://jubo.otakusdream.com/downloads/Jubo%20Classics%20%5bKB+PAD%5d%20(001-150)%20%5bMain%5d%20(Beta2017-Mar06).zip)
    * [(Jubo) 151-300](https://jubo.otakusdream.com/downloads/Jubo%20New%20Era%20%5bKB+PAD%5d%20(151-300)%20%5bMain%5d%20(Beta2017-Mar06).zip)
    * [(Jubo) 301-450](https://jubo.otakusdream.com/downloads/Jubo%20Impulsion%20%5bKB+PAD%5d%20(301-450)%5b2018Dec01%5d%5bMain%5d%5b0DF8017A%5d.zip)
    * [(Jubo) 451-500](https://jubo.otakusdream.com/downloads/Jubo%20Cessation%20+%202019%20%5bKB+PAD%5d%20(451-500)%20%5b2020Mar31%5d%7bMain%7d.zip)
    * [(Jubo) 2020](https://jubo.otakusdream.com/downloads/Jubo%202020%20Simfiles%20%5b2021Jan05%5d%5bDDD3E28F%5d.zip)
1. Navigate to `scripts/`
1. Parse `.sm` files to JSON: `./all.sh ./smd_1_extract.sh`
1. Filter JSON files (removing mines, etc.): `./all.sh ./smd_2_filter.sh`
1. Split dataset 80/10/10: `./all.sh ./smd_3_dataset.sh`
1. Analyze dataset (e.g.): `./smd_4_analyze.sh fraxtil`
