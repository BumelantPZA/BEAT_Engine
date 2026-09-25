import json,sys
for n in sys.argv[1:]:
    try: r=json.load(open(f"{n}_r2.json"))
    except FileNotFoundError: print(n,"missing"); continue
    fr=r["frequencies"]; t=lambda k: sum(f["timings"].get(k,0) for f in fr)/len(fr)
    print(f"{n:12s} wall/f {r['wall_s']/len(fr):.2f}  bem_op {t('bem_operator_s'):.2f}  fem_cond {t('fem_condensation_s'):.2f} (fact {t('fem_condensation_factorization_s'):.2f})  bemmat {t('bem_matrix_s'):.2f} block {t('block_assembly_s'):.2f}  elim {t('interface_elimination_s'):.2f}  cfact {t('coupled_factorization_s'):.2f}  field {t('field_s'):.2f}")
