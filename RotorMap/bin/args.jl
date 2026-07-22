# args() = "serv"

# args() = "convert -i ref.bin -o ref3.fasta"
# args() = " convert -o refc.bin -i ref.fasta"

# args() = "convert -i reads.bin -o reads.fasta"

# args() = "convert -i readsn50kl100ke10.fasta -o reads.bin"



# args() = "gen ref 1000000000 -o ref.fasta"
# args() = "gen ref 1000000000 -o ref.bin"
# args() = "gen ref 100 -o ref.bin"


# args() = "gen reads -n 50000 -k 100000 -i ref.fasta -o readsn50kl100ke10.bin --err 0.1"
# args() = "gen reads -n 10000 -k 100000 -i ref.fasta -o readsn10kl100ke30.fasta --err 0.3"

# args() = "gen reads -n 10000 -k 100000 -i ref.bin -o readsn10kl100ke01.bin --err 0.01"
# args() = "gen reads -n 10000 -k 100000 -i ref.bin -o readsn10kl100ke05.bin --err 0.05"
# args() = "gen reads -n 10000 -k 100000 -i ref.bin -o readsn10kl100ke25.bin --err 0.25"
# args() = "gen reads -n 10000 -k 100000 -i ref.bin -o readsn10kl100ke20.bin --err 0.2"
# args() = "gen reads -n 10000 -k 100000 -i ref.bin -o readsn10kl100ke15.bin --err 0.15"
# args() = "gen reads -n 10000 -k 100000 -i ref.bin -o readsn10kl100ke10.bin --err 0.1"
# args() = "gen reads -n 10000 -k 100000 -i ref.bin -o readsn10kl100ke30.bin --err 0.3"
# args() = "gen reads -n 10000 -k 100000 -i ref.bin -o reads.fasta"
# args() = " gen reads -n 50 -k 100000 -i ref.bin -o reads.bin"
# args() = "gen reads -n 100000 -k 100000 -i ref100k.bin"
# args() = "index -i ref.bin -k 100000 -s 3 -o indexs3.bin"
# args() = " index -i ref.fasta -k 100000"
# args() = "index -i ref.bin -k 100000 -s 5 -o index.bin"
# args() = "index -i ref.fasta -k 100000 -s 5 -o index.bin"

# args() = "index -i ref.fasta -k 100000 -s 3 -o indexs3.bin --err 0.1"

# args() = "map -i indexs4.bin -r reads.bin"

# args() = "map -r readsn50kl100ke10_v2.bin -i indexs3.bin -s5 -m16 "
# args() = "map -r readsn50kl100ke10.fasta -i indexs3.bin -s5 -m16 "
# args() = "map -r reads.bin -i indexs3.bin -s5 -m16 "

# args() = "  map -r readsn10kl100ke10.bin -i index.bin"
# args() = "map -r readsn10kl100ke10.bin -i indexs3.bin            "
# args() = "map -r readsn10kl100ke10.bin -s8 -m1    "

args() = " map -r readsn10kl100ke10.bin"

# args() = "map -r readsn10kl100ke10.bin -i index.bin -s8 -m1"
 


# args() = " map -r readsn50kl100ke10.fasta -s6 -m4 -i index.bin"

# args() = "map -r readsn10kl100ke10.fasta"
# args() = "map -r readsn10kl100ke30.fasta"

# args() = "map -r reads.bin"
# args() = "map -r reads.fasta"
# args() = "map -r reads1.bin"

# args() = " serv exit"

# args() = "map -r reads1.bin -i index.bin -o pos.csv"

# args() = "map -r reads4.bin -o pos.csv"

