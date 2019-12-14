"""
    graph_diagnostic.jl
    
    #Output variables names:
        z
        u
        v
        w
        q_tot
        q_liq
        e_tot
        thd
        thl
        thv
        e_int
        h_m
        h_t
        qt_sgs
        ht_sgs
        vert_eddy_mass_flx
        vert_eddy_u_flx
        vert_eddy_v_flx
        vert_eddy_qt_flx
        vert_qt_flx
        vert_eddy_ql_flx
        vert_eddy_qv_flx
        vert_eddy_thd_flx
        vert_eddy_thv_flx
        vert_eddy_thl_flx
        uvariance
        vvariance
        wvariance
        wskew
        TKE
        SGS
        Ri

"""

using Plots; pyplot()
using VegaLite
using DataFrames, FileIO
using DelimitedFiles

include("./diagnostic_vars.jl")

function usage()
    println("""
Usage:
    julia graph_diagnostic.jl <diagnostic_file.jld2> <diagnostic_name>""")
end

function start(args::Vector{String})
    #data = load(args[1])

    #
    # USER INPUTS:
    #
    user_specified_time = 2347.654637996573 # set it to -1 if you want the plotter to detect and show the last time step data only
    time_average = "n"
    isimex = "y"
    if isimex == "yes" || isimex == "y"
        #data = load("../../output/GCLOUD/julia-sm/diagnostics-2019-12-14T16:48:05.227.jld2")
        data = load("../../output/GCLOUD/julia-sm/geostrophic-positive-sign/diagnostics-2019-12-12T16:27:38.751.jld2")
        integrator_method = string("1D IMEX ")
    else
        data = load("../../output/GCLOUD/julia-sm/diagnostics-2019-12-14T16:48:05.227.jld2")
        integrator_method = string("EXPL")
    end
    #data = load("../../output/diagnostics-2019-12-06T17:21:36.343.jld2")  #sphere
    
    out_vars = ["ht_sgs",
                "qt_sgs",
                "h_m",
                "h_t",
                "vert_eddy_qt_flx",
                "q_tot",
                "q_liq",
                "wvariance",
                "wskew",
                "thd",
                "thv",
                "thl",
                "w",
                "N",                
                "uvariance",
                "vvariance",
                "vert_eddy_thv_flx",
                "u",
                "v"]
    
    zvertical = 1200
    
    #
    # END USER INPUTS:
    #


    #
    # Stevens et al. 2005 measurements:
    #
    qt_stevens  = readdlm("../../output/Stevens2005Data/experimental_qt_stevens2005.csv", ',', Float64)
    ql_stevens  = readdlm("../../output/Stevens2005Data/experimental_ql_stevens2005.csv", ',', Float64)
    thl_stevens = readdlm("../../output/Stevens2005Data/experimental_thetal_stevens2005.csv", ',', Float64)
    tkelower_stevens = readdlm("../../output/Stevens2005Data/lower_limit_tke_time_stevens2005.csv", ',', Float64)
    tkeupper_stevens = readdlm("../../output/Stevens2005Data/upper_limit_tke_time_stevens2005.csv", ',', Float64)
#    ww_stevens = readdlm("../../output/Stevens2005Data/.csv", ',', Float64)
    
    @show keys(data)
    println("data for $(length(data)) time steps in file")
    
    diff = 100

    times = parse.(Float64,keys(data))

    if user_specified_time < 0   
        timeend = maximum(times)
    else
        timeend = user_specified_time
    end
    ntimes = length(times)
    @show(ntimes)
    
    time_data = string(timeend)
    Nqk = size(data[time_data], 1)
    nvertelem = size(data[time_data], 2)
    Z = zeros(Nqk * nvertelem)
    for ev in 1:nvertelem
        for k in 1:Nqk
            dv = diagnostic_vars(data[time_data][k,ev])
            Z[k+(ev-1)*Nqk] = dv.z
        end
    end

    V1  = zeros(Nqk * nvertelem)
    V2  = zeros(Nqk * nvertelem)
    V3  = zeros(Nqk * nvertelem)
    V4  = zeros(Nqk * nvertelem)
    V5  = zeros(Nqk * nvertelem)
    V6  = zeros(Nqk * nvertelem)
    V7  = zeros(Nqk * nvertelem)
    V8  = zeros(Nqk * nvertelem)
    V9  = zeros(Nqk * nvertelem)
    V10 = zeros(Nqk * nvertelem)
    V11 = zeros(Nqk * nvertelem)
    V12 = zeros(Nqk * nvertelem)
    V13 = zeros(Nqk * nvertelem)
    V14 = zeros(Nqk * nvertelem)
    V15 = zeros(Nqk * nvertelem)
    V16 = zeros(Nqk * nvertelem)
    V17 = zeros(Nqk * nvertelem)
    V18 = zeros(Nqk * nvertelem)
    V19 = zeros(Nqk * nvertelem)
    if time_average == "yes" || time_average == "y"

        time_str = string(integrator_method, ". Time averaged from 0 to ", ceil(timeend), " s")
        
        for key in keys(data)
            for ev in 1:nvertelem
                for k in 1:Nqk
                    dv = diagnostic_vars(data[key][k,ev])
                    V1[k+(ev-1)*Nqk] += getproperty(dv, Symbol(out_vars[1]))
                    V2[k+(ev-1)*Nqk] += getproperty(dv, Symbol(out_vars[2]))
                    V3[k+(ev-1)*Nqk] += getproperty(dv, Symbol(out_vars[3]))
                    V4[k+(ev-1)*Nqk] += getproperty(dv, Symbol(out_vars[4]))
                    V5[k+(ev-1)*Nqk] += getproperty(dv, Symbol(out_vars[5]))
                    V6[k+(ev-1)*Nqk] += getproperty(dv, Symbol(out_vars[6]))
                    V7[k+(ev-1)*Nqk] += getproperty(dv, Symbol(out_vars[7]))
                    V8[k+(ev-1)*Nqk] += getproperty(dv, Symbol(out_vars[8]))
                    V9[k+(ev-1)*Nqk] += getproperty(dv, Symbol(out_vars[9]))
                    V10[k+(ev-1)*Nqk] += getproperty(dv, Symbol(out_vars[10]))
                    V11[k+(ev-1)*Nqk] += getproperty(dv, Symbol(out_vars[11]))
                    V12[k+(ev-1)*Nqk] += getproperty(dv, Symbol(out_vars[12]))
                    V13[k+(ev-1)*Nqk] += getproperty(dv, Symbol(out_vars[13]))
                    V14[k+(ev-1)*Nqk] += getproperty(dv, Symbol(out_vars[14]))
                    V15[k+(ev-1)*Nqk] += getproperty(dv, Symbol(out_vars[15]))
                    V16[k+(ev-1)*Nqk] += getproperty(dv, Symbol(out_vars[16]))
                    V17[k+(ev-1)*Nqk] += getproperty(dv, Symbol(out_vars[17]))
                    V18[k+(ev-1)*Nqk] += getproperty(dv, Symbol(out_vars[18]))
                    V19[k+(ev-1)*Nqk] += getproperty(dv, Symbol(out_vars[19]))
                end
            end
        end
    else

        key = time_data #this is a string
        time_str = string(integrator_method, ". Instantaneous at t= ", ceil(timeend), " s")
        
        ntimes = 1
        for ev in 1:nvertelem
            for k in 1:Nqk
                dv = diagnostic_vars(data[key][k,ev])
                V1[k+(ev-1)*Nqk] += getproperty(dv, Symbol(out_vars[1]))
                V2[k+(ev-1)*Nqk] += getproperty(dv, Symbol(out_vars[2]))
                V3[k+(ev-1)*Nqk] += getproperty(dv, Symbol(out_vars[3]))
                V4[k+(ev-1)*Nqk] += getproperty(dv, Symbol(out_vars[4]))
                V5[k+(ev-1)*Nqk] += getproperty(dv, Symbol(out_vars[5]))
                V6[k+(ev-1)*Nqk] += getproperty(dv, Symbol(out_vars[6]))
                V7[k+(ev-1)*Nqk] += getproperty(dv, Symbol(out_vars[7]))
                V8[k+(ev-1)*Nqk] += getproperty(dv, Symbol(out_vars[8]))
                V9[k+(ev-1)*Nqk] += getproperty(dv, Symbol(out_vars[9]))
                V10[k+(ev-1)*Nqk] += getproperty(dv, Symbol(out_vars[10]))
                V11[k+(ev-1)*Nqk] += getproperty(dv, Symbol(out_vars[11]))
                V12[k+(ev-1)*Nqk] += getproperty(dv, Symbol(out_vars[12]))
                V13[k+(ev-1)*Nqk] += getproperty(dv, Symbol(out_vars[13]))
                V14[k+(ev-1)*Nqk] += getproperty(dv, Symbol(out_vars[14]))
                V15[k+(ev-1)*Nqk] += getproperty(dv, Symbol(out_vars[15]))
                V16[k+(ev-1)*Nqk] += getproperty(dv, Symbol(out_vars[16]))
                V17[k+(ev-1)*Nqk] += getproperty(dv, Symbol(out_vars[17]))
                V18[k+(ev-1)*Nqk] += getproperty(dv, Symbol(out_vars[18]))
                V19[k+(ev-1)*Nqk] += getproperty(dv, Symbol(out_vars[19]))
            end
        end
    end

    p1 = plot(V1/ntimes, Z,
              linewidth=3,
              xaxis=("ht_sgs", (0, 250), 0:25:250),
              yaxis=("Altitude[m]", (0, zvertical)),
              label=("ht_sgs"),
              )

    p2 = plot(V2/ntimes, Z,
              linewidth=3,
              xaxis=("qt_sgs", (-5e-5, 2e-5)),
              yaxis=("Altitude[m]", (0, zvertical)),
              label=("qt_sgs"),
              )

    labels = ["h_m = e_i + gz + RmT" "h_t = e_t + RmT"]
    p3 = plot([V3/ntimes V4/ntimes], Z,
              linewidth=3,
              xaxis=("Moist and total enthalpies"), #(1.08e8, 1.28e8), 1.08e8:0.1e8:1.28e8),
              yaxis=("Altitude[m]", (0, zvertical)),
              label=labels,
              )
    
    pwqt = plot(V5*1e+3/ntimes, Z,
              linewidth=3,
              xaxis=("<w qt> (m/s g/kg)"), #(0, 0.), 0:0.:1),
              yaxis=("Altitude[m]", (0, zvertical)),
              label=("<w qt>"),
              )

    pqt = plot(V6*1e+3/ntimes, Z,
              linewidth=3,
              xaxis=("<qt>", (0, 12), 0:2:12),
              yaxis=("Altitude[m]", (0, zvertical)),
              label=("<qt>"),
              )

    qt_rf01 = qt_stevens[:,1]
    z_rf01  = qt_stevens[:,2]
    p5 = plot!(qt_rf01,z_rf01,seriestype=:scatter,
               markersize = 10,
               markercolor = :black,
               label=("<qt experimental>"))
    ##
    
    pql = plot(V7*1e+3/ntimes, Z,
              linewidth=3,
              xaxis=("<ql>", (-0.05, 0.5), -0.05:0.1:0.5),
              yaxis=("Altitude[m]", (0, zvertical)),
              label=("<ql>"),
              )

    ql_rf01 = ql_stevens[:,1]
    z_rf01  = ql_stevens[:,2]
    p6 = plot!(ql_rf01,z_rf01,seriestype=:scatter,
               markersize = 10,
               markercolor = :black,
               label=("<ql experimental>"))

    puu = plot(V15/ntimes, Z,
              linewidth=3,
              xaxis=("<u'u'>"),# (-0.1, 0.6), -0.1:0.1:0.6),
              yaxis=("Altitude[m]", (0, zvertical)),
              label=("<u'u'>"),
               )

    pvv = plot(V16/ntimes, Z,
              linewidth=3,
              xaxis=("<v'v'>"),# (-0.1, 0.6), -0.1:0.1:0.6),
              yaxis=("Altitude[m]", (0, zvertical)),
              label=("<v'v'>"),
              )



    pww = plot(V8/ntimes, Z,
              linewidth=3,
              xaxis=("<w'w'>"),# (-0.1, 0.6), -0.1:0.1:0.6),
              yaxis=("Altitude[m]", (0, zvertical)),
              label=("<w'w'>"),
              )

  tke = 0.5*(V8.*V8 + V15.*V15 + V16.*V16)
  ptke = plot(tke/ntimes^2, Z,
              linewidth=3,
              xaxis=("TKE"),# (-0.1, 0.6), -0.1:0.1:0.6),
              yaxis=("Altitude[m]", (0, zvertical)),
              label=("<TKE>"),
              )

    tke_rf01 = tkelower_stevens[:,1]
    z_rf01   = tkelower_stevens[:,2]
   # ptke = plot!(tke_rf01,z_rf01,seriestype=:scatter,
   #            markersize = 10,
   #            markercolor = :black,
   #            label=("<lower limit tke experimental>"))

    
    pwww = plot(V9/ntimes, Z,
              linewidth=3,
              xaxis=("<w'w'w'>"), #, (-0.15, 0.15), -0.15:0.05:0.15),
              yaxis=("Altitude[m]", (0, zvertical)),
              label=("<w'w'w'>"),
              )
    
    data = [V10/ntimes V11/ntimes V12/ntimes]
    labels = ["θ" "θv" "θl"]
    pthl = plot(data, Z,
              linewidth=3,
              xaxis=("<θ>"), #, (-0.15, 0.15), -0.15:0.05:0.15),
              yaxis=("Altitude[m]", (0, zvertical)),
              label=labels,
               )

    thl_rf01 = thl_stevens[:,1]
    z_rf01  = thl_stevens[:,2]
    pthl = plot!(thl_rf01,z_rf01,seriestype=:scatter,
               markersize = 10,
               markercolor = :black,
              label=("<thl experimental>"))

    pth = plot(data, Z,
              linewidth=3,
              xaxis=("<θ>"), #, (-0.15, 0.15), -0.15:0.05:0.15),
              yaxis=("Altitude[m]", (0, zvertical)),
              label=labels,
              )
##

    B = 9.81*V17/289.0
    pB = plot(B/ntimes, Z,
              linewidth=3,
              xaxis=("g<w'θv>/θ_0"), #, (-0.15, 0.15), -0.15:0.05:0.15),
              yaxis=("Altitude[m]", (0, zvertical)),
              label=("g<w'θv>/θ_0"),
               )


    pu = plot(V18/ntimes, Z,
              linewidth=3,
              xaxis=("u", (0, 10), 0:2:10),
              yaxis=("Altitude[m]", (0, zvertical)),
              label=("u (m/s)"),
              )

    pv = plot(V19/ntimes, Z,
              linewidth=3,
              xaxis=("v", (-10, 0), -10:2:0),
              yaxis=("Altitude[m]", (0, zvertical)),
              label=("v (m/s)"),
              )

    pw = plot(V13/ntimes, Z,
              linewidth=3,
              xaxis=("w"), #, (-0.15, 0.15), -0.15:0.05:0.15),
              yaxis=("Altitude[m]", (0, zvertical)),
              label=("w (m/s)"),
               )

    p11 = plot(V14/ntimes, Z,
              linewidth=3,
              xaxis=("dθv/dz"), #, (-0.15, 0.15), -0.15:0.05:0.15),
              yaxis=("Altitude[m]", (0, zvertical)),
              label=("dθv/dz"),
              )

    
    f=font(14,"courier")
    #plot(pqt, pql, pth, ptke, pww, pwww, layout = (2,3), titlefont=f, tickfont=f, legendfont=f, guidefont=f, title=time_str)
    plot(pu,  pv,  pthl, pth,
         pqt, pql, pwqt, pB,
         puu, pww, pwww, ptke, layout = (3,4), titlefont=f, tickfont=f, legendfont=f, guidefont=f, title=time_str)

    plot!(size=(900,1200))
end

#if length(ARGS) != 3 || !endswith(ARGS[1], ".jld2")
#    usage()
#end
start(ARGS)
