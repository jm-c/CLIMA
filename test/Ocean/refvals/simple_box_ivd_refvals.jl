    refVals=[]
    refPrecs=[]

# SC ========== Test number 1 reference values and precision match template. =======
# SC ========== /home/jmc/cliMa/cliMa_new_jmc/test/Ocean/SplitExplicit/simple_box_2dt.jl test reference values ======================================
# BEGIN SCPRINT
# varr - reference values (from reference run)
# parr - digits match precision (hand edit as needed)
#
# [
#  [ MPIStateArray Name, Field Name, Maximum, Minimum, Mean, Standard Deviation ],
#  [         :                :          :        :      :          :           ],
# ]
varr = [
 [  "oce Q_3D",   "u[1]", -1.50962361753736729053e-01,  1.73787194589605287209e-01,  4.68098176666432563148e-03,  1.95918702087896703934e-02 ],
 [  "oce Q_3D",   "u[2]", -1.41693277014211693743e-01,  1.46256903927605874660e-01, -2.51527289903215496569e-03,  1.75577307352173875299e-02 ],
 [  "oce Q_3D",       :η, -7.42525448756723882582e-01,  3.69952349991580053956e-01, -2.12300579216294864728e-04,  2.72196784419864945548e-01 ],
 [  "oce Q_3D",       :θ,  4.64352816670868251414e-03,  9.93366845172767831684e+00,  2.49647184360694174288e+00,  2.18037711611593776340e+00 ],
 [   "oce aux",       :w, -1.93642118079277015088e-04,  1.64388494599151930133e-04,  4.30134987337706410355e-07,  1.44731816723112416059e-05 ],
 [   "oce aux",    :pkin, -9.02450734444008029200e-01,  0.00000000000000000000e+00, -3.32591606145584972598e-01,  2.55055561435138855586e-01 ],
 [   "oce aux",     :wz0, -2.13130590203792726805e-05,  3.03764295307291628048e-05, -1.29351489286388951209e-10,  7.79133385457225507325e-06 ],
 [   "oce aux", "u_d[1]", -1.47678995808530610923e-01,  1.16175707854701368293e-01, -3.40162514353149251730e-05,  1.18326667626959050605e-02 ],
 [   "oce aux", "u_d[2]", -1.41297097563442025647e-01,  1.40410041592192808002e-01, -8.66960554174275520279e-06,  1.16477781860160160832e-02 ],
 [   "oce aux", "ΔGu[1]", -1.82462180035313031949e-06,  1.75730579970564754281e-06, -3.58905562801251356298e-08,  2.63341240419211030793e-07 ],
 [   "oce aux", "ΔGu[2]", -1.23605953229188910373e-06,  2.21985915696832233044e-06,  1.31099407764425287042e-06,  6.56785741293597450970e-07 ],
 [   "oce aux",       :y,  0.00000000000000000000e+00,  4.00000000000000046566e+06,  2.00000000000000000000e+06,  1.15573163901915703900e+06 ],
 [ "baro Q_2D",   "U[1]", -1.28633175515245561371e+01,  6.10318317172543416405e+01,  4.70740761170336963204e+00,  1.49615297140118084229e+01 ],
 [ "baro Q_2D",   "U[2]", -2.82520952983562132488e+01,  6.47619508637900196391e+01, -2.47935794730033354227e+00,  1.34335968079767127392e+01 ],
 [ "baro Q_2D",       :η, -7.42731313350159738640e-01,  3.69996965336395311486e-01, -2.12333995604797106261e-04,  2.72347141148035598590e-01 ],
 [  "baro aux",  "Gᵁ[1]", -1.75730579970564764446e-03,  1.82462180035313035337e-03,  3.58905562801251301771e-05,  2.63354276791606873361e-04 ],
 [  "baro aux",  "Gᵁ[2]", -2.21985915696832226268e-03,  1.23605953229188902411e-03, -1.31099407764425279418e-03,  6.56818254634440362900e-04 ],
 [  "baro aux", "U_c[1]", -1.28413474945461345555e+01,  6.09612852044860531464e+01,  4.71376723747856551938e+00,  1.49474605677143106419e+01 ],
 [  "baro aux", "U_c[2]", -2.82757649462843900778e+01,  6.47162674795010843809e+01, -2.50650840410464237351e+00,  1.34316239381179141077e+01 ],
 [  "baro aux",     :η_c, -7.42525448756723882582e-01,  3.69952349991580053956e-01, -2.12300579216321237946e-04,  2.72210259174677282612e-01 ],
 [  "baro aux", "U_s[1]", -1.28415899595582310155e+01,  6.09623267104856623178e+01,  4.71378782923472794408e+00,  1.49476602108240328448e+01 ],
 [  "baro aux", "U_s[2]", -2.82766214528259105521e+01,  6.47161195504508555132e+01, -2.50657419606653197874e+00,  1.34316536525727965312e+01 ],
 [  "baro aux",     :η_s, -7.42533259412902157948e-01,  3.69956621986456846152e-01, -2.12295179177931456769e-04,  2.72211064871600239012e-01 ],
 [  "baro aux",  "Δu[1]", -4.52584773336763403293e-04,  4.01399962579148425305e-04, -1.17814846783320024927e-05,  1.01681656567342138366e-04 ],
 [  "baro aux",  "Δu[2]", -2.43941531101083075800e-04,  3.86401032853338948295e-04,  5.11421508443407068253e-05,  7.52601602185662730366e-05 ],
 [  "baro aux",  :η_diag, -7.39952355750089774133e-01,  3.68437192305505589740e-01, -2.11882685629188936925e-04,  2.71995932052603417439e-01 ],
 [  "baro aux",      :Δη, -3.57682889898169875664e-03,  4.28352100974604965700e-03, -4.17893587097645250512e-07,  1.12668312540401493148e-03 ],
 [  "baro aux",       :y,  0.00000000000000000000e+00,  4.00000000000000046566e+06,  2.00000000000000000000e+06,  1.15578885204060329124e+06 ],
]
parr = [
 [  "oce Q_3D",   "u[1]",    12,    12,    12,    12 ],
 [  "oce Q_3D",   "u[2]",    12,    12,    12,    12 ],
 [  "oce Q_3D",       :η,    12,    12,     8,    12 ],
 [  "oce Q_3D",       :θ,    12,    12,    12,    12 ],
 [   "oce aux",       :w,    12,    12,     8,    12 ],
 [   "oce aux",    :pkin,    12,    12,    12,    12 ],
 [   "oce aux",     :wz0,    12,    12,     8,    12 ],
 [   "oce aux", "u_d[1]",    12,    12,    12,    12 ],
 [   "oce aux", "u_d[2]",    12,    12,    12,    12 ],
 [   "oce aux", "ΔGu[1]",    12,    12,    12,    12 ],
 [   "oce aux", "ΔGu[2]",    12,    12,    12,    12 ],
 [   "oce aux",       :y,    12,    12,    12,    12 ],
 [ "baro Q_2D",   "U[1]",    12,    12,    12,    12 ],
 [ "baro Q_2D",   "U[2]",    12,    12,    12,    12 ],
 [ "baro Q_2D",       :η,    12,    12,     8,    12 ],
 [  "baro aux",  "Gᵁ[1]",    12,    12,    12,    12 ],
 [  "baro aux",  "Gᵁ[2]",    12,    12,    12,    12 ],
 [  "baro aux", "U_c[1]",    12,    12,    12,    12 ],
 [  "baro aux", "U_c[2]",    12,    12,    12,    12 ],
 [  "baro aux",     :η_c,    12,    12,     8,    12 ],
 [  "baro aux", "U_s[1]",    12,    12,    12,    12 ],
 [  "baro aux", "U_s[2]",    12,    12,    12,    12 ],
 [  "baro aux",     :η_s,    12,    12,     8,    12 ],
 [  "baro aux",  "Δu[1]",    12,    12,    12,    12 ],
 [  "baro aux",  "Δu[2]",    12,    12,    12,    12 ],
 [  "baro aux",  :η_diag,    12,    12,     8,    12 ],
 [  "baro aux",      :Δη,    12,    12,     8,    12 ],
 [  "baro aux",       :y,    12,    12,    12,    12 ],
]
# END SCPRINT
# SC ====================================================================================

    append!(refVals ,[ varr ] )
    append!(refPrecs,[ parr ] )

#! format: on