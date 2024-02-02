# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 10:55:55 2023

@author: ge84tin
"""

import sys
import os
import mph
#import jdk
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET

class DDPGEnv():
    def __init__(self):
        self.size= 30
        self.lst= list(range(3,902))#np.linspace(3,902)
        #self.state= np.random.choice([0, 1], size=(1,900), p=[0, 1])
        #self.action_space= round(np.random.uniform(0, 1),2)  # action space round(np.random.uniform(0, 1),2) 
        

    def ComSim(self,idx1_list, idx2_list):
        lst = list(range(2,902))
        # lst= lst.tolist()
        client= mph.start(cores=1)
        model= client.create('Model')
        model.clear()
        model=model.java

        model.modelNode().create("comp1")
        #odel.geom().create("geom1")
        model.param().set("F", "30[cm]");
        model.param().set("Gamma", "1.4");
        model.param().set("epsilonP0", "0.95");
        model.param().set("Rf0", "8900[Pa/(s*m^2)]");
        model.param().set("Lv0", "180e-6[m]");
        model.param().set("Lth0", "360e-6[m]");
        model.param().set("Tau0", "1.42");
        model.param().set("rho0", "1.2[kg/m^3]");
        model.param().set("K0", "141e5[Pa]");
        model.param().set("rho1", "7800 [kg/m^3]");
        model.param().set("K1", "6e11[Pa]");
        model.param().set("eta", "1.81e-5[Pa*s]");
        model.param().set("P0", "101e3[Pa]");
        model.param().set("fstart", "300[Hz]");
        model.param().set("fstop", "3000[Hz]");

        model.component("comp1").geom().create("geom1", 2);
        model.component("comp1").mesh().create("mesh1");
        model.result().table().create("tbl1", "Table");

        model.geom("geom1").lengthUnit("cm");
        model.geom("geom1").create("r1", "Rectangle");
        model.geom("geom1").feature("r1").set("pos", ["0", "0"]);
        model.geom("geom1").feature("r1").set("size", ["3", "3"]);
        model.geom("geom1").create("r2", "Rectangle");
        model.geom("geom1").feature("r2").set("pos", ["-20", "0"]);
        model.geom("geom1").feature("r2").set("size", ["20", "3"]);
        model.geom("geom1").create("pol1", "Polygon");
        model.geom("geom1").feature("pol1").set("type", "open");
        model.geom("geom1").feature("pol1").set("source", "table");
        model.geom("geom1").feature("pol1").set("table", [["0.1", "3"], ["0.1", "0"]]);
        model.geom("geom1").create("pol2", "Polygon");
        model.geom("geom1").feature("pol2").set("type", "open");
        model.geom("geom1").feature("pol2").set("source", "table");
        model.geom("geom1").feature("pol2").set("table", [["0", "2.9"], ["3", "2.9"]]);
        model.geom("geom1").create("arr1", "Array");
        model.geom("geom1").feature("arr1").set("type", "linear");
        model.geom("geom1").feature("arr1").set("linearsize", "29");
        model.geom("geom1").feature("arr1").set("displ", ["0.1", "0"]);
        model.geom("geom1").feature("arr1").selection("input").set("pol1");
        model.geom("geom1").create("arr2", "Array");
        model.geom("geom1").feature("arr2").set("type", "linear");
        model.geom("geom1").feature("arr2").set("linearsize", "29");
        model.geom("geom1").feature("arr2").set("displ", ["0", "-0.1"]);
        model.geom("geom1").feature("arr2").selection("input").set("pol2");
        model.geom("geom1").run("fin");
        #
        model.component("comp1").variable().create("var1");
        model.component("comp1").variable("var1").set("k0", "2*pi*freq/343[m/s]");
        model.component("comp1").variable("var1").set("rhoP", "(Tau0*rho0/epsilonP0)*(1 + (Rf0*epsilonP0/(1i*omega*rho0*Tau0))*sqrt(1+ (1i*4*Tau0^2 *eta *rho0*omega/(Rf0^2 * Lv0^2 * epsilonP0^2))))");
        model.component("comp1").variable("var1").set("KP", "(Gamma*P0/epsilonP0)*(Gamma - (Gamma -1)*(1- (1i*8*eta/(Lth0^2*Pr*rho0*omega))*sqrt(1 + (1i*Lth0^2 * Pr *rho0*omega/(16*eta))))^-1 )^-1");
        model.component("comp1").variable("var1").set("Pr", "0.7", "Prandtl number");
        model.component("comp1").variable("var1").set("omega", "2*pi*acpr.freq", "Angular frequency");
        model.component("comp1").variable("var1").set("rho_ef", "rho0 + (rhoP - rho0)*1");
        model.component("comp1").variable("var1").set("K_ef", "K0 + (KP - K0)*1");

        model.component("comp1").material().create("mat1", "Common");
        model.component("comp1").material().create("mat2", "Common");
        model.component("comp1").material("mat1").propertyGroup("def").func().create("eta", "Piecewise");
        model.component("comp1").material("mat1").propertyGroup("def").func().create("Cp", "Piecewise");
        model.component("comp1").material("mat1").propertyGroup("def").func().create("rho", "Analytic");
        model.component("comp1").material("mat1").propertyGroup("def").func().create("k", "Piecewise");
        model.component("comp1").material("mat1").propertyGroup("def").func().create("cs", "Analytic");
        model.component("comp1").material("mat1").propertyGroup("def").func().create("an1", "Analytic");
        model.component("comp1").material("mat1").propertyGroup("def").func().create("an2", "Analytic");
        model.component("comp1").material("mat1").propertyGroup().create("RefractiveIndex", "Refractive index");
        model.component("comp1").material("mat1").propertyGroup().create("NonlinearModel", "Nonlinear model");
        model.component("comp1").material("mat1").propertyGroup().create("idealGas", "Ideal gas");
        model.component("comp1").material("mat1").propertyGroup("idealGas").func().create("Cp", "Piecewise");
        model.component("comp1").material("mat2").selection().set(idx1_list);

        model.component("comp1").physics().create("acpr", "PressureAcoustics", "geom1");
        model.component("comp1").physics("acpr").create("pom1", "PoroacousticsModel", 2);
        model.component("comp1").physics("acpr").feature("pom1").selection().set(idx2_list);
        model.component("comp1").physics("acpr").create("fpam3", "FrequencyPressureAcousticsModel", 2);
        model.component("comp1").physics("acpr").feature("fpam3").selection().set(idx1_list);
        model.component("comp1").physics("acpr").create("port1", "Port", 1);
        model.component("comp1").physics("acpr").feature("port1").selection().set(1);

        model.component("comp1").mesh("mesh1").create("ftri2", "FreeTri");
        model.component("comp1").mesh("mesh1").feature("ftri2").create("size1", "Size");
        model.component("comp1").mesh("mesh1").feature("ftri2").feature("size1").selection().geom("geom1", 2);
        model.component("comp1").mesh("mesh1").feature("ftri2").feature("size1").selection().set(lst.insert(0,1));

        model.component("comp1").probe().create("var1", "GlobalVariable");
        model.result().table("tbl1").label("Probe Table 1");
        model.component("comp1").view("view1").axis().set("xmin", -0.6869015693664551);
        model.component("comp1").view("view1").axis().set("xmax", 4.514914512634277);
        model.component("comp1").view("view1").axis().set("ymin", -0.033447228372097015);
        model.component("comp1").view("view1").axis().set("ymax", 3.020456552505493);

        model.component("comp1").material("mat1").label("Air");
        model.component("comp1").material("mat1").set("family", "air");
        model.component("comp1").material("mat1").propertyGroup("def").func("eta").set("arg", "T");
        model.component("comp1").material("mat1").propertyGroup("def").func("eta").set("pieces", ["200.0", "1600.0", "-8.38278E-7+8.35717342E-8*T^1-7.69429583E-11*T^2+4.6437266E-14*T^3-1.06585607E-17*T^4"]);
        model.component("comp1").material("mat1").propertyGroup("def").func("eta").set("argunit", "K");
        model.component("comp1").material("mat1").propertyGroup("def").func("eta").set("fununit", "Pa*s");
        model.component("comp1").material("mat1").propertyGroup("def").func("Cp").set("arg", "T");
        model.component("comp1").material("mat1").propertyGroup("def").func("Cp").set("pieces", ["200.0", "1600.0", "1047.63657-0.372589265*T^1+9.45304214E-4*T^2-6.02409443E-7*T^3+1.2858961E-10*T^4"]);
        model.component("comp1").material("mat1").propertyGroup("def").func("Cp").set("argunit", "K");
        model.component("comp1").material("mat1").propertyGroup("def").func("Cp").set("fununit", "J/(kg*K)");
        model.component("comp1").material("mat1").propertyGroup("def").func("rho").set("expr", "pA*0.02897/R_const[K*mol/J]/T");
        model.component("comp1").material("mat1").propertyGroup("def").func("rho").set("args", ["pA", "T"]);
        model.component("comp1").material("mat1").propertyGroup("def").func("rho").set("argunit", "Pa,K");
        model.component("comp1").material("mat1").propertyGroup("def").func("rho").set("fununit", "kg/m^3");
        model.component("comp1").material("mat1").propertyGroup("def").func("rho").set("plotargs", [["pA", "0", "1"], ["T", "0", "1"]]);
        model.component("comp1").material("mat1").propertyGroup("def").func("k").set("arg", "T");
        model.component("comp1").material("mat1").propertyGroup("def").func("k").set("pieces", ["200.0", "1600.0", "-0.00227583562+1.15480022E-4*T^1-7.90252856E-8*T^2+4.11702505E-11*T^3-7.43864331E-15*T^4"]);
        model.component("comp1").material("mat1").propertyGroup("def").func("k").set("argunit", "K");
        model.component("comp1").material("mat1").propertyGroup("def").func("k").set("fununit", "W/(m*K)");
        model.component("comp1").material("mat1").propertyGroup("def").func("cs").set("expr", "sqrt(1.4*R_const[K*mol/J]/0.02897*T)");
        model.component("comp1").material("mat1").propertyGroup("def").func("cs").set("args", "T");
        model.component("comp1").material("mat1").propertyGroup("def").func("cs").set("argunit", "K");
        model.component("comp1").material("mat1").propertyGroup("def").func("cs").set("fununit", "m/s");
        model.component("comp1").material("mat1").propertyGroup("def").func("cs").set("plotargs", ["T", "273.15", "373.15"]);
        model.component("comp1").material("mat1").propertyGroup("def").func("an1").set("funcname", "alpha_p");
        model.component("comp1").material("mat1").propertyGroup("def").func("an1").set("expr", "-1/rho(pA,T)*d(rho(pA,T),T)");
        model.component("comp1").material("mat1").propertyGroup("def").func("an1").set("args", ["pA", "T"]);
        model.component("comp1").material("mat1").propertyGroup("def").func("an1").set("argunit", "Pa,K");
        model.component("comp1").material("mat1").propertyGroup("def").func("an1").set("fununit", "1/K");
        model.component("comp1").material("mat1").propertyGroup("def").func("an1").set("plotargs", [["pA", "101325", "101325"], ["T", "273.15", "373.15"]]);
        model.component("comp1").material("mat1").propertyGroup("def").func("an2").set("funcname", "muB");
        model.component("comp1").material("mat1").propertyGroup("def").func("an2").set("expr", "0.6*eta(T)");
        model.component("comp1").material("mat1").propertyGroup("def").func("an2").set("args", "T");
        model.component("comp1").material("mat1").propertyGroup("def").func("an2").set("argunit", "K");
        model.component("comp1").material("mat1").propertyGroup("def").func("an2").set("fununit", "Pa*s");
        model.component("comp1").material("mat1").propertyGroup("def").func("an2").set("plotargs", ["T", "200", "1600"]);
        model.component("comp1").material("mat1").propertyGroup("def").set("thermalexpansioncoefficient", "null");
        model.component("comp1").material("mat1").propertyGroup("def").set("molarmass", "null");
        model.component("comp1").material("mat1").propertyGroup("def").set("bulkviscosity", "null");
        model.component("comp1").material("mat1").propertyGroup("def").set("thermalexpansioncoefficient", ["alpha_p(pA,T)", "0", "0", "0", "alpha_p(pA,T)", "0", "0", "0", "alpha_p(pA,T)"]);
        model.component("comp1").material("mat1").propertyGroup("def").set("molarmass", "0.02897[kg/mol]");
        model.component("comp1").material("mat1").propertyGroup("def").set("bulkviscosity", "muB(T)");
        model.component("comp1").material("mat1").propertyGroup("def").set("relpermeability", ["1", "0", "0", "0", "1", "0", "0", "0", "1"]);
        model.component("comp1").material("mat1").propertyGroup("def").set("relpermittivity", ["1", "0", "0", "0", "1", "0", "0", "0", "1"]);
        model.component("comp1").material("mat1").propertyGroup("def").set("dynamicviscosity", "eta(T)");
        model.component("comp1").material("mat1").propertyGroup("def").set("ratioofspecificheat", "1.4");
        model.component("comp1").material("mat1").propertyGroup("def").set("electricconductivity", ["0[S/m]", "0", "0", "0", "0[S/m]", "0", "0", "0", "0[S/m]"]);
        model.component("comp1").material("mat1").propertyGroup("def").set("heatcapacity", "Cp(T)");
        model.component("comp1").material("mat1").propertyGroup("def").set("density", "rho(pA,T)");
        model.component("comp1").material("mat1").propertyGroup("def").set("thermalconductivity", ["k(T)", "0", "0", "0", "k(T)", "0", "0", "0", "k(T)"]);
        model.component("comp1").material("mat1").propertyGroup("def").set("soundspeed", "cs(T)");
        model.component("comp1").material("mat1").propertyGroup("def").addInput("temperature");
        model.component("comp1").material("mat1").propertyGroup("def").addInput("pressure");
        model.component("comp1").material("mat1").propertyGroup("RefractiveIndex").set("n", "null");
        model.component("comp1").material("mat1").propertyGroup("RefractiveIndex").set("ki", "null");
        model.component("comp1").material("mat1").propertyGroup("RefractiveIndex").set("n", "null");
        model.component("comp1").material("mat1").propertyGroup("RefractiveIndex").set("ki", "null");
        model.component("comp1").material("mat1").propertyGroup("RefractiveIndex").set("n", "null");
        model.component("comp1").material("mat1").propertyGroup("RefractiveIndex").set("ki", "null");
        model.component("comp1").material("mat1").propertyGroup("RefractiveIndex").set("n", ["1", "0", "0", "0", "1", "0", "0", "0", "1"]);
        model.component("comp1").material("mat1").propertyGroup("RefractiveIndex").set("ki", ["0", "0", "0", "0", "0", "0", "0", "0", "0"]);
        model.component("comp1").material("mat1").propertyGroup("NonlinearModel").set("BA", "(def.gamma+1)/2");
        model.component("comp1").material("mat1").propertyGroup("idealGas").func("Cp").label("Piecewise 2");
        model.component("comp1").material("mat1").propertyGroup("idealGas").func("Cp").set("arg", "T");
        model.component("comp1").material("mat1").propertyGroup("idealGas").func("Cp").set("pieces", ["200.0", "1600.0", "1047.63657-0.372589265*T^1+9.45304214E-4*T^2-6.02409443E-7*T^3+1.2858961E-10*T^4"]);
        model.component("comp1").material("mat1").propertyGroup("idealGas").func("Cp").set("argunit", "K");
        model.component("comp1").material("mat1").propertyGroup("idealGas").func("Cp").set("fununit", "J/(kg*K)");
        model.component("comp1").material("mat1").propertyGroup("idealGas").set("Rs", "R_const/Mn");
        model.component("comp1").material("mat1").propertyGroup("idealGas").set("heatcapacity", "Cp(T)");
        model.component("comp1").material("mat1").propertyGroup("idealGas").set("ratioofspecificheat", "1.4");
        model.component("comp1").material("mat1").propertyGroup("idealGas").set("molarmass", "0.02897");
        model.component("comp1").material("mat1").propertyGroup("idealGas").addInput("temperature");
        model.component("comp1").material("mat1").propertyGroup("idealGas").addInput("pressure");
        model.component("comp1").material("mat2").propertyGroup("def").set("density", "rho1");
        model.component("comp1").material("mat2").propertyGroup("def").set("soundspeed", "sqrt(K1/rho1)");

        model.component("comp1").physics("acpr").feature("pom1").set("FluidModel", "JohnsonChampouxAllard");
        model.component("comp1").physics("acpr").feature("pom1").set("epsilon_p_mat", "userdef");
        model.component("comp1").physics("acpr").feature("pom1").set("epsilon_p", 0.95);
        model.component("comp1").physics("acpr").feature("pom1").set("Lv_mat", "userdef");
        model.component("comp1").physics("acpr").feature("pom1").set("Lv", "180e-6");
        model.component("comp1").physics("acpr").feature("pom1").set("Lth_mat", "userdef");
        model.component("comp1").physics("acpr").feature("pom1").set("Lth", "360e-6");
        model.component("comp1").physics("acpr").feature("pom1").set("tau_mat", "userdef");
        model.component("comp1").physics("acpr").feature("pom1").set("tau", 1.42);
        model.component("comp1").physics("acpr").feature("pom1").set("Rf_mat", "userdef");
        model.component("comp1").physics("acpr").feature("pom1").set("Rf", "8900");
        model.component("comp1").physics("acpr").feature("fpam3").set("LinearElasticOption", "Krho");
        model.component("comp1").physics("acpr").feature("fpam3").set("rho_mat", "userdef");
        model.component("comp1").physics("acpr").feature("fpam3").set("rho", "rho1");
        model.component("comp1").physics("acpr").feature("fpam3").set("K_eff_mat", "userdef");
        model.component("comp1").physics("acpr").feature("fpam3").set("K_eff", "K1");
        model.component("comp1").physics("acpr").feature("fpam3").label("Pressure Acoustics: Solid Mat");
        model.component("comp1").physics("acpr").feature("port1").set("PortType", "Slit");
        model.component("comp1").physics("acpr").feature("port1").set("pamp", "1");


        model.component("comp1").mesh("mesh1").feature("size").set("hauto", "2");
        model.component("comp1").mesh("mesh1").feature("size").set("custom", "on");
        model.component("comp1").mesh("mesh1").feature("size").set("hmax", "0.54");
        model.component("comp1").mesh("mesh1").feature("size").set("hmin", "0.00203");
        model.component("comp1").mesh("mesh1").feature("ftri2").feature("size1").set("hauto", "2");
        model.component("comp1").mesh("mesh1").feature("ftri2").feature("size1").set("custom", "on");
        model.component("comp1").mesh("mesh1").feature("ftri2").feature("size1").set("hmax", "0.1");
        model.component("comp1").mesh("mesh1").feature("ftri2").feature("size1").set("hmaxactive", "on");
        model.component("comp1").mesh("mesh1").feature("ftri2").feature("size1").set("hmin", "0.00203");
        model.component("comp1").mesh("mesh1").feature("ftri2").feature("size1").set("hminactive", "on");
        model.component("comp1").mesh("mesh1").feature("ftri2").feature("size1").set("hgradactive", "on");
        model.component("comp1").mesh("mesh1").run();

        model.component("comp1").probe("var1").set("expr", "1- abs(acpr.S11)^2");
        model.component("comp1").probe("var1").set("unit", "");
        model.component("comp1").probe("var1").set("descr", "1- abs(acpr.S11)^2");
        model.component("comp1").probe("var1").set("table", "tbl1");
        model.component("comp1").probe("var1").set("window", "window1");

        model.study().create("std1");
        model.study("std1").create("freq", "Frequency");

        model.sol().create("sol1");
        model.sol("sol1").study("std1");
        model.sol("sol1").attach("std1");
        model.sol("sol1").create("st1", "StudyStep");
        model.sol("sol1").create("v1", "Variables");
        model.sol("sol1").create("s1", "Stationary");
        model.sol("sol1").feature("s1").create("p1", "Parametric");
        model.sol("sol1").feature("s1").create("fc1", "FullyCoupled");
        model.sol("sol1").feature("s1").feature().remove("fcDef");

        model.result().dataset().create("dset2", "Solution");
        model.result().dataset("dset2").set("probetag", "var1");
        model.result().numerical().create("gev1", "EvalGlobal");
        model.result().numerical("gev1").set("data", "dset2");
        model.result().numerical("gev1").set("probetag", "var1");
        model.result().create("pg1", "PlotGroup2D");
        model.result().create("pg2", "PlotGroup2D");
        model.result().create("pg3", "PlotGroup1D");
        model.result("pg1").create("surf1", "Surface");
        model.result("pg2").create("surf1", "Surface");
        model.result("pg2").feature("surf1").set("expr", "acpr.Lp_t");
        model.result("pg3").set("probetag", "window1");
        model.result("pg3").create("tblp1", "Table");
        model.result("pg3").feature("tblp1").set("probetag", "var1");

        model.component("comp1").probe("var1").genResult("none");
        model.result("pg3").tag("pg3");
        model.study("std1").feature("freq").set("plist", "range(fstart,100,fstop)");

        model.sol("sol1").attach("std1");
        model.sol("sol1").feature("st1").label("Compile Equations: Frequency Domain");
        model.sol("sol1").feature("v1").label("Dependent Variables 1.1");
        model.sol("sol1").feature("v1").set("clistctrl", ["p1"]);
        model.sol("sol1").feature("v1").set("cname", ["freq"]);
        model.sol("sol1").feature("v1").set("clist", ["range(fstart,100,fstop)"]);
        model.sol("sol1").feature("s1").label("Stationary Solver 1.1");
        model.sol("sol1").feature("s1").feature("dDef").label("Direct 1");
        model.sol("sol1").feature("s1").feature("aDef").label("Advanced 1");
        model.sol("sol1").feature("s1").feature("aDef").set("complexfun", "on");
        model.sol("sol1").feature("s1").feature("p1").label("Parametric 1.1");
        model.sol("sol1").feature("s1").feature("p1").set("pname", ["freq"]);
        model.sol("sol1").feature("s1").feature("p1").set("plistarr", ["range(fstart,100,fstop)"]);
        model.sol("sol1").feature("s1").feature("p1").set("punit", ["Hz"]);
        model.sol("sol1").feature("s1").feature("p1").set("pcontinuationmode", "no");
        model.sol("sol1").feature("s1").feature("p1").set("preusesol", "auto");
        model.sol("sol1").feature("s1").feature("fc1").label("Fully Coupled 1.1");
        model.sol("sol1").runAll();


        model.result().dataset("dset2").label("Probe Solution 2");
        model.result().numerical("gev1").set("unit", ["1"]);
        model.result("pg1").label("Acoustic Pressure (acpr)");
        model.result("pg1").feature("surf1").set("colortable", "Wave");
        model.result("pg1").feature("surf1").set("colortablesym", "on");
        model.result("pg1").feature("surf1").set("resolution", "normal");
        model.result("pg2").label("Sound Pressure Level (acpr)");
        model.result("pg2").feature("surf1").set("resolution", "normal");
        model.result("pg3").set("xlabel", "freq (Hz)");
        tablearray=model.result("pg3").set("ylabel", "1- abs(acpr.S11)<sup>2</sup>");
        model.result("pg3").set("xlabelactive", "off");
        model.result("pg3").set("ylabelactive", "off");
        table_str=model.result().table('tbl1').getTableData(1);
        table_str= np.array(table_str, dtype=object)

        # observ=[state0, table_str[1]]
        #model.save('InputPython2')
        client.remove('Model')
        return table_str  #two observations /absorption/ binary state
    
    def Bin2Ind(self,state):
        self.state= np.reshape(self.state, (1,900))
        lst = list(range(2,902))
        lst=np.asarray(lst)
        #P= np.random.uniform(0, 1) # action space
        #state0=np.random.choice([0, 1], size=(1,900), p=[P, 1-P]) # state space


        idx1=np.where(state == -2)[1]
        idx2=np.where(state == 2)[1]


        idx1_list = lst[idx1]
        idx2_list = lst[idx2]


        #idx1_list.insert(0, 1)


        return idx1_list, idx2_list
    
    
    def step(self, action):
        self.state= np.random.choice([-2, 2], size=(1,900), p=[action/900, 1-(action/900)])
        self.state= np.reshape(self.state, (1,900))
        idx1_list, idx2_list=self.Bin2Ind(self.state)
        table_str=self.ComSim( idx1_list, idx2_list)

            
        absorption = [str(i) for i in table_str[:,1]]
        absorpSum= sum(float(i) for i in absorption)
        absp= [float(i) for i in absorption]
        fracvoid= np.sum((self.state==2))
            
#        if absorpSum > 23:
#            reward= 500
#        elif absorpSum > 20:
#            reward= 400
#        elif absorpSum > 19:
#            reward= 100
#        else:
#            reward= -100
        diff=26 - absorpSum
        reward= abs((1- diff/26)**(2)) 
        done = absorpSum >= 26 

        
        return np.array(self.state), reward, done, absorpSum, absp
        
    def reset(self):
        self.state = np.random.choice([-2, 2], size=(1,900), p=[0.1, 0.9])
        return self.state
    
