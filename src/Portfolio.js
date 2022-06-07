import React, { Component } from 'react';

import Slider from 'rc-slider';
import { Row, Col } from 'react-bootstrap';

import 'rc-slider/assets/index.css';
import PortfolioSection from './PortfolioSection';

const portfolioData = require('./portfolio.json');

class Portfolio extends Component {
  constructor(props) {
    super(props);
    this.state = {
      data: portfolioData,
      slider: 2,
    };
  }

  render() {
    const { data, slider } = this.state;

    return (
      <main>
        <section id="portfolio">
          <h2 className="section-heading">things i&apos;ve done</h2>
          <center>
            <Slider
              defaultValue={2}
              min={1}
              max={4}
              marks={{
                1: 'Less things',
                2: 'Default',
                3: 'More things',
                4: 'Shit List',
              }}
              dots
              onChange={(v) => this.setState({ slider: v })}
            />
          </center>
          <div className="portfolio-box">
            <Row>
              <Col xs={12} md={6} className="item">
                <PortfolioSection
                  data={data.experience}
                  title="Employment"
                  slider={slider}
                />
                <PortfolioSection
                  data={data.events}
                  title="Participation"
                  slider={slider}
                />
                <PortfolioSection
                  data={data.stuff}
                  title="Misc"
                  slider={slider}
                />
              </Col>
              <Col xs={12} md={6} className="item">
                <PortfolioSection
                  data={data.hobbies}
                  title="Hobbies"
                  slider={slider}
                />
                <PortfolioSection
                  data={data.projects}
                  title="Projects"
                  slider={slider}
                />
                {slider > 2
                  && (
                    <PortfolioSection
                      data={data.awards}
                      title="Awards"
                      slider={slider}
                    />
                  )}
              </Col>
            </Row>
          </div>
        </section>
      </main>
    );
  }
}


export default Portfolio;
