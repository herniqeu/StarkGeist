import clsx from 'clsx';
import Heading from '@theme/Heading';
import styles from './styles.module.css';

type FeatureItem = {
  title: string;
  Svg: React.ComponentType<React.ComponentProps<'svg'>>;
  description: JSX.Element;
};

const FeatureList: FeatureItem[] = [
  {
    title: 'Fácil Implementação',
    Svg: require('@site/static/img/undraw_docusaurus_mountain.svg').default,
    description: (
      <>
        Nossa ferramenta foi projetada para se integrar perfeitamente ao seu fluxo de trabalho, permitindo que você comece a tomar decisões estratégicas com dados precisos de forma rápida e eficiente.      
      </>
    ),
  },
  {
    title: 'Foque no que é importante.',
    Svg: require('@site/static/img/undraw_docusaurus_tree.svg').default,
    description: (
      <>
        Deixe nossa ferramenta cuidar da análise de dados e dos cálculos enquanto você se concentra no que realmente importa: estratégia e crescimento.
      </>
    ),
  },
  {
    title: 'Impulsionado por IA.',
    Svg: require('@site/static/img/undraw_docusaurus_react.svg').default,
    description: (
      <>
        Aproveite o poder da inteligência artificial para simular cenários "what if" e gerar previsões financeiras e econômicas detalhadas, apoiando decisões estratégicas de alto impacto      
      </>
    ),
  },
];

function Feature({title, Svg, description}: FeatureItem) {
  return (
    <div className={clsx('col col--4')}>
      <div className="text--center">
        <Svg className={styles.featureSvg} role="img" />
      </div>
      <div className="text--center padding-horiz--md">
        <Heading as="h3">{title}</Heading>
        <p>{description}</p>
      </div>
    </div>
  );
}

export default function HomepageFeatures(): JSX.Element {
  return (
    <section className={styles.features}>
      <div className="container">
        <div className="row">
          {FeatureList.map((props, idx) => (
            <Feature key={idx} {...props} />
          ))}
        </div>
      </div>
    </section>
  );
}
