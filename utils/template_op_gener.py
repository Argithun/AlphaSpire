import csv, json
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
OP_DIR = BASE / "data" / "wq_operators"
IN_CSV = OP_DIR / "operators.csv"

TEMP_DIR = BASE / "data" / "wq_template_operators"
TEMP_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV = TEMP_DIR / "template_operators.csv"
OUT_JSON = TEMP_DIR / "template_operators.json"

TYPE_MAP = {
    "Arithmetic:NAry": {"add","multiply","max","min"},
    "Arithmetic:Binary": {"subtract","divide","power","signed_power"},
    "Arithmetic:Unary": {"sign","abs","log","sqrt","inverse","reverse","tanh","sigmoid"},
    "Logical:Unary": {"is_nan","not"},
    "Logical:Binary": {"and","or","less","equal","greater","not_equal","less_equal","greater_equal"},
    "Conditional": {"if_else","trade_when"},
    "TS:Aggregation": {"ts_mean","ts_sum","ts_std_dev","ts_product","ts_av_diff","ts_min_diff","ts_zscore","ts_skewness","ts_entropy","ts_count_nans"},
    "TS:WindowIndex": {"ts_arg_max","ts_arg_min","kth_element","last_diff_value","days_from_last_change","ts_step"},
    "TS:CorrelationRegression": {"ts_corr","ts_covariance","ts_regression"},
    "TS:Transform": {"ts_backfill","ts_delay","ts_decay_linear","ts_target_tvr_decay","ts_scale","ts_quantile","ts_rank","ts_delta","ts_min_max_cps","ts_min_max_diff","hump"},
    "CrossSection:Standardize": {"winsorize","rank","zscore","scale","normalize","quantile"},
    "CrossSection:RegressionProj": {"regression_proj"},
    "Group:Aggregation": {"bucket","densify","group_mean","group_rank","group_extra","group_backfill","group_scale","group_zscore","group_neutralize","group_cartesian_product"},
    "Vector:LinearAlgebra": {"vector_proj","vector_neut","vec_sum","vec_avg"},
    "Special:Domain": {"inst_pnl"},
}


def generate_template_ops():
    # invert for lookups
    op_to_type = {}
    for t,s in TYPE_MAP.items():
        for op in s:
            op_to_type[op] = t

    rows = []
    unknown = set()
    if not IN_CSV.exists():
        raise FileNotFoundError(f"{IN_CSV} not found.")

    with open(IN_CSV,newline="",encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            name = (r.get("name") or "").strip()
            t = op_to_type.get(name, "UNKNOWN")
            # assign signature templates by type
            signature = ""
            if t == "Arithmetic:NAry":
                signature = "NAry(x1,x2,...[,options])"
            elif t == "Arithmetic:Binary":
                signature = "Binary(x,y[,options])"
            elif t == "Arithmetic:Unary":
                signature = "Unary(x[,options])"
            elif t == "Logical:Unary":
                signature = "LogicalUnary(x)"
            elif t == "Logical:Binary":
                signature = "LogicalBinary(a,b)"
            elif t == "Conditional":
                signature = "Conditional(cond, a, b) or Conditional(cond, value)"
            elif t == "TS:Aggregation":
                signature = "TS_Agg(x,d[,options])"
            elif t == "TS:WindowIndex":
                signature = "TS_Index(x,d[,k])"
            elif t == "TS:CorrelationRegression":
                signature = "TS_Bivariate(y,x,d[,options])"
            elif t == "TS:Transform":
                signature = "TS_Transform(x,d[,params])"
            elif t == "CrossSection:Standardize":
                signature = "CS_Std(x[,params])"
            elif t == "CrossSection:RegressionProj":
                signature = "regression_proj(y,x)"
            elif t == "Group:Aggregation":
                signature = "GroupOp(x, group[,params])"
            elif t == "Vector:LinearAlgebra":
                signature = "VectorOp(x[,y])"
            elif t == "Special:Domain":
                signature = "Special(...)"
            else:
                signature = ""

            if t == "UNKNOWN":
                unknown.add(name)
            r_out = dict(r)
            r_out["type"] = t
            r_out["signature_template"] = signature
            rows.append(r_out)

    # write CSV
    if rows:
        fieldnames = list(rows[0].keys())
        with open(OUT_CSV,"w",newline="",encoding="utf-8") as f:
            writer = csv.DictWriter(f,fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        with open(OUT_JSON,"w",encoding="utf-8") as f:
            json.dump(rows,f,ensure_ascii=False,indent=2)

    print("Wrote", OUT_CSV, OUT_JSON)
    if unknown:
        print("Unknown operators to classify manually:", sorted(unknown))


if __name__ == "__main__":
    generate_template_ops()