import React, { Component } from 'react';
import PropTypes from 'prop-types';

import { CSSTransition } from 'react-transition-group';

import 'rc-slider/assets/index.css';

class PortfolioSection extends Component {
  constructor(props) {
    super(props);
    this.state = {
      data: props.data,
      title: props.title,
      slider: props.slider,
    };
  }

  componentDidUpdate(prevProps) {
    const { slider } = this.state;
    if (slider !== prevProps.slider) {
      this.setState({
        slider,
      });
    }
  }

  render() {
    const { data, title, slider } = this.state;

    return (
      <div className="portfolio-section">
        <h1>{title}</h1>
        {
          data.map((item) => {
            const showItem = item.toggle.indexOf(slider) !== -1;
            const points = item.points.map((p) => {
              const showPoint = p.toggle.indexOf(slider) !== -1;
              return (
                <CSSTransition
                  classNames="points"
                  appear
                  in={showPoint}
                  unmountOnExit
                  mountOnEnter
                  timeout={{ enter: 1500, exit: 300 }}
                >
                  <li dangerouslySetInnerHTML={{ __html: p.content }} />
                </CSSTransition>
              );
            });
            return (
              <CSSTransition
                classNames="points"
                appear
                in={showItem}
                unmountOnExit
                mountOnEnter
                timeout={{ enter: 1500, exit: 300 }}
              >
                <div className="portfolio-item">
                  <div className="header">
                    <span className="title">{item.thing}</span>
                    <span className="divider-strong">&nbsp;&#x2F;&#x2F;&nbsp;</span>
                    <span className="desc">{item.description}</span>
                    <span className="divider-weak">&nbsp;&#x2F;&#x2F;&nbsp;</span>
                    <span className="date">{item.date}</span>
                  </div>
                  <ul>
                    {points}
                  </ul>
                </div>
              </CSSTransition>
            );
          })
        }
      </div>
    );
  }
}

PortfolioSection.propTypes = {
  data: PropTypes.object.isRequired,
  title: PropTypes.string.isRequired,
  slider: PropTypes.number.isRequired,
};

export default PortfolioSection;
