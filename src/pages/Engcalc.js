import React, { Component } from 'react';
import './styles/Engcalc.css';
import { Tex } from 'react-tex';
import * as math from 'mathjs';

class Engcalc extends Component {
  constructor(props) {
    super(props);

    this.equation = '';
    this.answerEquation = '';
    this.eState = true;
    this.answer = '';
  }

  componentDidMount() {
    this.play();

    const script = document.createElement('script');
    script.type = 'text/javascript';
    script.async = true;
    script.onload = function () {
      // remote script has loaded
    };
    script.src = `${process.env.PUBLIC_URL}engcalc/engcalc.js`;
    document.getElementsByTagName('head')[0].appendChild(script);
  }

  _handleKeyDown = (event) => {
    if (/^[0-9()+-/*coslogsin%tan^]$/.test(event.key)) {
      this.click(event.key);
    }
    if (/^Backspace$/.test(event.key)) {
      this.backspace(event.key);
    }
    if (/^Enter$/.test(event.key)) {
      this.evaluate(event.key);
    }
  };

  click(operation) {
    this.equation += operation;
    this.eState = true;
  }

  clear() {
    this.equation = '';
    this.eState = true;
  }

  backspace() {
    this.equation = this.equation.slice(0, this.equation.length - 1);
    this.eState = true;
  }

  evaluate() {
    let toEval = this.equation;
    toEval = toEval.replace('x', '*');
    const lnm = toEval.match(/ln\((.+)\)/);
    const logm = toEval.match(/log\((.+)\)/);
    if (lnm) {
      toEval = toEval.replace(/ln\((.+)\)/, `log(${lnm[1]}, e)`);
    }
    if (logm) {
      toEval = toEval.replace(/log\((.+)\)/, `log(${logm[1]}, 10)`);
    }

    try {
      const lahtech = window.Module.UTF8ToString(window.Module.ccall('run', null, ['string'], [math.evaluate(toEval).toString()]));
      this.answerEquation = `${toEval} = ${lahtech}`;
      this.equation = '';
      this.eState = false;
    } catch {
      alert('beep beep bad equation beep');
    }
  }

  render() {
    return (
      <div className="Engcalc">
        <div className="toprow">
          {this.eState
            && (
              <div className="equation">
                {this.equation}
              </div>
            )}
          {!this.eState
            && (
              <div className="equation">
                <Tex texContent={this.answerEquation} />
              </div>
            )}
        </div>
        <div className="row">
          <img src={`${process.env.PUBLIC_URL}/engcalc/1title.gif`} alt="" />
          <img src={`${process.env.PUBLIC_URL}/engcalc/2factorial.gif`} onClick={() => this.click('!')} alt="" />
          <img src={`${process.env.PUBLIC_URL}/engcalc/3openbracket.gif`} onClick={() => this.click('(')} alt="" />
          <img src={`${process.env.PUBLIC_URL}/engcalc/4closingbracket.gif`} onClick={() => this.click(')')} alt="" />
          <img src={`${process.env.PUBLIC_URL}/engcalc/5modulo.gif`} onClick={() => this.click('%')} alt="" />
          <img src={`${process.env.PUBLIC_URL}/engcalc/6AC.gif`} onClick={() => this.clear()} alt="" />
        </div>
        <div className="row">
          <img src={`${process.env.PUBLIC_URL}/engcalc/7music.png`} onClick={() => this.music()} alt="" />
          <img src={`${process.env.PUBLIC_URL}/engcalc/8sin.gif`} onClick={() => this.click('sin(')} alt="" />
          <img src={`${process.env.PUBLIC_URL}/engcalc/9ln.gif`} onClick={() => this.click('ln(')} alt="" />
          <img src={`${process.env.PUBLIC_URL}/engcalc/101.gif`} onClick={() => this.click('1')} alt="" />
          <img src={`${process.env.PUBLIC_URL}/engcalc/113.gif`} onClick={() => this.click('3')} alt="" />
          <img src={`${process.env.PUBLIC_URL}/engcalc/125.gif`} onClick={() => this.click('5')} alt="" />
          <img src={`${process.env.PUBLIC_URL}/engcalc/13:.gif.gif`} onClick={() => this.click('/')} alt="" />
        </div>
        <div className="row">
          <img src={`${process.env.PUBLIC_URL}/engcalc/14pi.gif`} onClick={() => this.click('pi')} alt="" />
          <img src={`${process.env.PUBLIC_URL}/engcalc/15cos.gif`} onClick={() => this.click('cos(')} alt="" />
          <img src={`${process.env.PUBLIC_URL}/engcalc/16log.gif`} onClick={() => this.click('log(')} alt="" />
          <img src={`${process.env.PUBLIC_URL}/engcalc/177.gif`} onClick={() => this.click('7')} alt="" />
          <img src={`${process.env.PUBLIC_URL}/engcalc/189.gif`} onClick={() => this.click('9')} alt="" />
          <img src={`${process.env.PUBLIC_URL}/engcalc/190.gif`} onClick={() => this.click('0')} alt="" />
          <img src={`${process.env.PUBLIC_URL}/engcalc/20x.gif`} onClick={() => this.click('x')} alt="" />
        </div>
        <div className="row">
          <img src={`${process.env.PUBLIC_URL}/engcalc/21e.gif`} onClick={() => this.click('e')} alt="" />
          <img src={`${process.env.PUBLIC_URL}/engcalc/22tan.gif`} onClick={() => this.click('tan(')} alt="" />
          <img src={`${process.env.PUBLIC_URL}/engcalc/23sqrt.gif`} onClick={() => this.click('sqrt(')} alt="" />
          <img src={`${process.env.PUBLIC_URL}/engcalc/242.gif`} onClick={() => this.click('2')} alt="" />
          <img src={`${process.env.PUBLIC_URL}/engcalc/254.gif`} onClick={() => this.click('4')} alt="" />
          <img src={`${process.env.PUBLIC_URL}/engcalc/266.gif`} onClick={() => this.click('6')} alt="" />
          <img src={`${process.env.PUBLIC_URL}/engcalc/27-.gif`} onClick={() => this.click('-')} alt="" />
        </div>
        <div className="row">
          <img src={`${process.env.PUBLIC_URL}/engcalc/28ans.gif`} onClick={() => this.click(this.answer)} alt="" />
          <img src={`${process.env.PUBLIC_URL}/engcalc/29exp.png`} onClick={() => this.click('*10^')} alt="" />
          <img src={`${process.env.PUBLIC_URL}/engcalc/30^.gif`} onClick={() => this.click('^')} alt="" />
          <img src={`${process.env.PUBLIC_URL}/engcalc/318.gif`} onClick={() => this.click('8')} alt="" />
          <img src={`${process.env.PUBLIC_URL}/engcalc/32..gif`} onClick={() => this.click('.')} alt="" />
          <img src={`${process.env.PUBLIC_URL}/engcalc/33=.png`} onClick={() => this.evaluate()} alt="" />
          <img src={`${process.env.PUBLIC_URL}/engcalc/34+.gif`} onClick={() => this.click('+')} alt="" />
        </div>
        <div className="row">
          Built by Howard Halim, William Zhao and Carol Chen
          <br />
          Constants:
          <ul>
            <li>Pi (3.14)</li>
            <li>Phi(1.618)</li>
            <li>e(2.718)</li>
            <li>Gamma, Euler–Mascheroni constant(0.577)</li>
            <li>Alpha, Fine-structure constant (0.007297)</li>
          </ul>
        </div>
      </div>
    );
  }
}

export default Engcalc;
